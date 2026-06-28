import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Union, Optional

#
class Base_Degradation_Manifold(nn.Module):
    """
    ARNIQA-style projector using existing feature maps (no ResNet).
    Args:
      in_channels: list of per-level channels (e.g., [256, 512, 1024])
      embedding_dim: output embedding D
      reduce_to: common channel size after 1x1 reduction (e.g., 256)
      pool_type: "gap" or "attn"
      fuse_type: "concat" or "mean"
      use_norm: L2-normalize f and g
      dropout_p: dropout on pooled vectors
    Forward:
      x: List[Tensor] of shape [B, Ci, Hi, Wi] or a single Tensor
    Returns:
      f: fused pre-projection feature vector [B, F]
      g: embedding [B, D]
    """
    def __init__(
        self,
        in_channels: List[int],   # per-level/layer input channels (e.g., [256, 512, 1024])
        embedding_dim: int = 128,
        reduce_to: int = 256,
        pool_type: str = "gap",
        fuse_type: str = "concat",
        use_norm: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert pool_type in ("gap", "attn")
        assert fuse_type in ("concat", "mean")

        self.use_norm = use_norm
        self.embedding_dim = embedding_dim
        self.in_channels = list(in_channels)
        self.reduce_to = int(reduce_to)
        self.pool_type = pool_type
        self.fuse_type = fuse_type

        # Per-level reduction to common size
        self.reducers = nn.ModuleList([
            Conv1x1(cin, self.reduce_to, use_norm=True) for cin in in_channels
        ])

        # Optional attention pooling
        if pool_type == "attn":
            self.poolers = nn.ModuleList([AttentionPool2d(self.reduce_to) for _ in in_channels])
        else:
            self.poolers = nn.ModuleList([nn.Identity() for _ in in_channels])

        # Per-level vector norm + dropout
        self.ln = nn.ModuleList([nn.LayerNorm(self.reduce_to) for _ in in_channels])
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Projector on fused f (mirrors ARNIQA MLP)
        fused_dim = self._fused_dim()
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, embedding_dim),
        )

    def _pool(self, x: torch.Tensor, pooler: nn.Module) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, C]
        if self.pool_type == "gap":
            return torch.mean(x, dim=(2, 3))
        elif self.pool_type == "attn":
            return pooler(x)
        else:
            raise ValueError(f"Unknown pool_type={self.pool_type}")

    def _fused_dim(self) -> int:
        if self.fuse_type == "concat":
            return self.reduce_to * len(self.in_channels)
        else:  # mean
            return self.reduce_to

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]):
        if isinstance(x, torch.Tensor):
            x = [x]
        assert len(x) == len(self.in_channels), \
            f"Expected {len(self.in_channels)} feature levels, got {len(x)}"

        per_level_vecs = []
        for xi, red, pooler, ln in zip(x, self.reducers, self.poolers, self.ln):
            hi = red(xi)                  # [B, reduce_to, H, W]
            vi = self._pool(hi, pooler)   # [B, reduce_to]
            vi = ln(vi)
            vi = self.drop(vi)
            per_level_vecs.append(vi)

        if self.fuse_type == "concat":
            f = torch.cat(per_level_vecs, dim=1)         # [B, reduce_to * L]
        else:
            f = torch.stack(per_level_vecs, dim=1).mean(dim=1)  # [B, reduce_to]

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            g = F.normalize(g, dim=1)

        return f, g


class Conv1x1(nn.Module):
    def __init__(self, cin: int, cout: int, use_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, bias=not use_norm)
        self.norm = nn.BatchNorm2d(cout) if use_norm else nn.Identity()
        self.act = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

class AttentionPool2d(nn.Module):
    """
    Learnable spatial pooling: produces a [B, C] vector from [B, C, H, W].
    """
    def __init__(self, c: int):
        super().__init__()
        self.score = nn.Conv2d(c, 1, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.score(x)                 # [B, 1, H, W]
        attn = attn.view(attn.shape[0], -1)  # [B, H*W]
        attn = torch.softmax(attn, dim=1)
        attn = attn.view(x.shape[0], 1, x.shape[2], x.shape[3])  # [B,1,H,W]
        pooled = (x * attn).sum(dim=(2,3))  # [B, C]
        return pooled



def init_arniqa(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Works well for SiLU/ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.running_mean.zero_()
            m.running_var.fill_(1.0)

        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Default for hidden layers (ReLU/SiLU)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Special-cases: attention score conv and final projector layer
    # Try to detect them by name or pass references explicitly.
    # Example: zero-init any Conv2d with out_channels==1 (attention score conv).
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == 1:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Example usage:
# model = ARNIQAFromFeatures(in_channels=[256,512,1024], embedding_dim=128, ...)
# init_arniqa(model)