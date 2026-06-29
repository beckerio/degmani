
# file: yolo_distortion_model_trainable.py

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from degmani.models.base_degmani import Base_Degradation_Manifold


def _is_detect_head(m: nn.Module) -> bool:
    # Ultralytics heads typically include Detect/Segment/Pose/etc.
    return m.__class__.__name__ in {"Detect", "Segment", "Pose", "RTDETRDecoder", "TaskAlignedHead"}


class Degradation_Manifold(nn.Module):
    """
    YOLO backbone + Base_Degradation_Manifold integration.
    Supports fine-tuning selected YOLO layers and contrastive forward that splits a concatenated batch.

    """
    def __init__(
        self,
        weights: str,
        layer_idxs: List[int],
        embedding_dim: int = 128,
        reduce_to: int = 256,
        pool_type: str = "gap",
        fuse_type: str = "concat",
        use_norm: bool = True,
        dropout_p: float = 0.0,
        imgsz: int = 640,
        train_backbone: bool = True,
        train_indices: Optional[List[int]] = None,  # which indices to train; None => train all except detect head
        freeze_detect_head: bool = True,
        device: Optional[str] = None,
        model_yaml: str = "",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not model_yaml:
            self.yolo = YOLO(weights).model.to(self.device)
        else:
            ckpt = torch.load(weights, map_location=self.device, weights_only=True)
            yolo_wrapper = YOLO(model_yaml)
            
            state_dict = ckpt['model_state_dict']
            
            if any(k.startswith('yolo.') for k in state_dict):
                state_dict = {k.removeprefix('yolo.'): v for k, v in state_dict.items()}
            
            yolo_wrapper.model.load_state_dict(state_dict, strict=False)
            self.yolo = yolo_wrapper.model.to(device)
            self.yolo.eval()
        
        self.layer_idxs = list(layer_idxs)
        self._feats: Dict[int, torch.Tensor] = {}
        self._hooks = []
        for i in self.layer_idxs:
            h = self.yolo.model[i].register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

        # Probe channel counts with a dummy pass (no grad)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device)
            _ = self.yolo(dummy)
            in_channels = []
            for i in self.layer_idxs:
                if i not in self._feats:
                    #print(F"feat {self._feats[i].shape[1]}")
                    raise RuntimeError(f"No feature captured for layer {i}. Check indices.")
                in_channels.append(self._feats[i].shape[1])
            self._feats.clear()

        # Build manifold head
        self.head = Base_Degradation_Manifold(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            reduce_to=reduce_to,
            pool_type=pool_type,
            fuse_type=fuse_type,
            use_norm=use_norm,
            dropout_p=dropout_p,
        ).to(device)

        # Set trainable/frozen params
        self._set_trainability(train_backbone, train_indices, freeze_detect_head)

    def _make_hook(self, idx: int):
        def fn(module, inp, out):
            if isinstance(out, (list, tuple)):
                out = next((t for t in out if isinstance(t, torch.Tensor)), out[0])
            self._feats[idx] = out  # no detach: gradients flow through
        return fn

    def _set_trainability(self, train_backbone: bool, train_indices: Optional[List[int]], freeze_detect_head: bool):
        # First freeze everything
        for p in self.yolo.parameters():
            p.requires_grad_(False)

        if train_backbone:
            if train_indices is None:
                # Train all modules except detect head
                for i, m in enumerate(self.yolo.model):
                    if freeze_detect_head and _is_detect_head(m):
                        for p in m.parameters():
                            p.requires_grad_(False)
                    else:
                        for p in m.parameters():
                            p.requires_grad_(True)
            else:
                # Train only specified indices; optionally freeze detect head
                train_set = set(train_indices)
                for i, m in enumerate(self.yolo.model):
                    req = (i in train_set)
                    if freeze_detect_head and _is_detect_head(m):
                        req = False
                    for p in m.parameters():
                        p.requires_grad_(req)

        # Manifold head always trainable
        for p in self.head.parameters():
            p.requires_grad_(True)

    def parameters_backbone(self) -> List[nn.Parameter]:
        # Return only trainable backbone parameters
        params = []
        for i, m in enumerate(self.yolo.model):
            if any(p.requires_grad for p in m.parameters()):
                params += [p for p in m.parameters() if p.requires_grad]
        return params

    #def parameters_backbone(self) -> List[nn.Parameter]:
    #    # Return only trainable backbone parameters
    #    return [p for p in self.yolo.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normal forward; gradients propagate into YOLO layers we left trainable.
        _ = self.yolo(x)
        feats = [self._feats[i] for i in self.layer_idxs]
        self._feats.clear()
        f, g = self.head(feats)
        return f, g

    def forward_contrastive(self, x: torch.Tensor, hard_neg: bool = True):
        """
        x is a concatenated batch in the order:
          [A, B, A_ds, B_ds] if hard_neg=True
          [A, B]            if hard_neg=False
        Returns embeddings in the same pairing order.
        """
        _ = self.yolo(x)
        feats = [self._feats[i] for i in self.layer_idxs]
        self._feats.clear()

        nblocks = 4 if hard_neg else 2
        chunks_per_layer = [t.chunk(nblocks, dim=0) for t in feats]

        def block_feats(b: int) -> List[torch.Tensor]:
            # list of per-layer tensors for block b
            return [chunks[b] for chunks in chunks_per_layer]

        # Compute embeddings per block (pass list into head to respect per-layer pooling)
        _, emb_a     = self.head(block_feats(0))
        _, emb_b     = self.head(block_feats(1))

        if hard_neg:
            _, emb_a_ds = self.head(block_feats(2))
            _, emb_b_ds = self.head(block_feats(3))
            return emb_a, emb_b, emb_a_ds, emb_b_ds

        return emb_a, emb_b

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []