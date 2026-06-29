from contextlib import nullcontext
from typing import OrderedDict
from typing import List, Dict, Any, Optional, Union

import torch
import torch.nn as nn

# for YOLO models
def capture_by_indices(net, indices, x, detach=True, to_cpu=False,
                       take_first_if_tuple=True, no_grad=True):
    values, handles = {}, []

    def make_hook(i):
        def hook(mod, inp, out):
            tensor = out[0] if take_first_if_tuple and isinstance(out, (list, tuple)) else out
            if detach:
                tensor = tensor.detach()
            if to_cpu:
                tensor = tensor.cpu()
            values[i] = tensor
        return hook

    for i in indices:
        handles.append(net.model[i].register_forward_hook(make_hook(i)))
    try:
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            _ = net(x)
    finally:
        for handle in handles:
            handle.remove()
    return values

 # Create a dictionary to store the outputs
class StyleExtractor(nn.Module):
    def __init__(self, model, style_layers):
        super(StyleExtractor, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.outputs = {}

        # Register hooks to extract features
        for name, layer in self.model.named_children():
            #print(f" {name} ")
            if name in self.style_layers:
                layer.register_forward_hook(self.save_output(name))

    def save_output(self, name):
        def hook(module, input, output):
            self.outputs[name] = output

        return hook

    def forward(self, x):
        self.outputs = {}  # Reset outputs
        _ = self.model(x)  # Forward pass
        return self.outputs


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {}
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        for layer_name in layers:
            getattr(self.model, layer_name).register_forward_hook(self.save_activation(layer_name))

    def save_activation(self, name):
        def hook(module, input, output):
            self._features[name] = output

        return hook

    def forward(self, x):
        _ = self.model(x)
        pooled_features = [self.global_avg_pool(self._features[layer]).squeeze(-1).squeeze(-1) for layer in self.layers]
        combined_features = torch.cat(pooled_features, dim=1)
        return nn.functional.normalize(combined_features, p=2, dim=1)  # L2 normalization


class FasterRCNNFeatureExtractor_alternat(nn.Module):
    def __init__(self, model, style_layers):
        super().__init__()
        self.model = model
        self.style_layers = set(style_layers)

        # Storage dicts
        self.backbone_features = {}
        self.roi_align_features = {}

        # Register backbone hooks for selected layers
        for name, module in self.model.backbone.body.named_children():
            if name in self.style_layers:
                module.register_forward_hook(self._backbone_hook(f"backbone_{name}"))

        # Register ROI Align hook ONCE (moved out of the loop)
        self.model.roi_heads.box_roi_pool.register_forward_hook(self._roi_align_hook)

    def _backbone_hook(self, name):
        def hook_fn(module, inputs, output):
            self.backbone_features[name] = output.detach()
        return hook_fn

    def _roi_align_hook(self, module, inputs, output):
        # inputs: (features, boxes, image_shapes)
        feat = inputs[0]
        if isinstance(feat, torch.Tensor):
            self.roi_align_features["roi_align_input"] = feat.detach()
        elif isinstance(feat, OrderedDict):
            self.roi_align_features["roi_align_input"] = OrderedDict(
                (k, v.detach()) for k, v in feat.items()
            )
        elif isinstance(feat, (list, tuple)):
            self.roi_align_features["roi_align_input"] = [
                x.detach() if isinstance(x, torch.Tensor) else x for x in feat
            ]
        else:
            self.roi_align_features["roi_align_input"] = feat
        return output  # must return output to not break the forward

    def forward(self, x):
        self.backbone_features.clear()
        self.roi_align_features.clear()
        _ = self.model(x)  # Forward pass through the detector
        return self.backbone_features, self.roi_align_features.get("roi_align_input")

 # Create a dictionary to store the outputs
class FasterRCNNFeatureExtractor(nn.Module):
    def __init__(self, model, style_layers):
        super(FasterRCNNFeatureExtractor, self).__init__()
        self.model = model
        self.style_layers = style_layers
        #self.outputs = {}

        # Storage dicts
        self.backbone_features = {}
        self.roi_align_features = {}

        for name, module in model.backbone.body.named_children():
            if name in style_layers:
                module.register_forward_hook(self.backbone_hook(f"backbone_{name}"))

        # Register ROI align hook — it's in model.roi_heads.box_roi_pool
        # Register ROI Align hook ONCE (moved out of the loop)
        model.roi_heads.box_roi_pool.register_forward_hook(self.roi_align_hook)

    def backbone_hook(self, name):
        def hook_fn(module, input, output):
            self.backbone_features[name] = output.detach()

        return hook_fn

        # Hook for ROI Align output (post-pooler)

    def roi_align_hook(self, module, input, output):
        roi_input = input[0]

        if isinstance(roi_input, torch.Tensor):
            self.roi_align_features["roi_align_input"] = roi_input.detach()
        elif isinstance(roi_input, OrderedDict):
            self.roi_align_features["roi_align_input"] = {
                k: v.detach() for k, v in roi_input.items()
            }
        elif isinstance(roi_input, (list, tuple)):
            self.roi_align_features["roi_align_input"] = [
                x.detach() if isinstance(x, torch.Tensor) else x for x in roi_input
            ]
        else:
            self.roi_align_features["roi_align_input"] = roi_input  # fallback

        return output



    def forward(self, x):
        self.backbone_features = {}  #
        self.roi_align_features = {}
        _ = self.model(x)  # Forward pass
        return self.backbone_features,  self.roi_align_features["roi_align_input"]


class YOLOActivationExtractor(nn.Module):
    """
    Activation extractor for Ultralytics YOLO (v5/v8/…).
    - style_layers: list of layer indices (from model.model) to hook.
    - capture_detect_input: if True, captures feature maps fed into Detect head (P3/P4/P5...).
    """
    def __init__(self,
                 yolo_model: nn.Module,
                 style_layers: Optional[List[int]] = None,
                 capture_detect_input: bool = True):
        super().__init__()
        # Accept either the Ultralytics wrapper (YOLO) or the raw Model
        self.core = getattr(yolo_model, "model", yolo_model)
        if not hasattr(self.core, "model"):
            raise ValueError("Provided model doesn't look like an Ultralytics YOLO Model (missing .model).")

        # ModuleList of layers
        self.layers = list(self.core.model)

        self.style_layers = style_layers or []
        self.capture_detect_input = capture_detect_input

        # Storage
        self.backbone_features: Dict[str, torch.Tensor] = {}
        self.detect_inputs: Dict[str, Any] = {}

        # Register hooks for requested indices
        for idx in self.style_layers:
            if idx < 0 or idx >= len(self.layers):
                raise IndexError(f"style_layers index {idx} out of range (0..{len(self.layers)-1}).")
            module = self.layers[idx]
            name = f"layer_{idx}_{module.__class__.__name__}"
            module.register_forward_hook(self._make_activation_hook(name))
        # Register hook to capture Detect head inputs (P3/P4/P5...)
        if self.capture_detect_input:
            for m in self.core.modules():
                if m.__class__.__name__.lower() == "detect":
                    m.register_forward_hook(self._detect_input_hook)
                    break
            else:
                # No Detect module found (e.g., segmentation/pose or custom head)
                self.capture_detect_input = False

    def _make_activation_hook(self, name: str):
        def hook(module, input, output):
            # Store detached activations
            try:
                self.backbone_features[name] = output.detach()
            except Exception:
                # Some modules may return tuples/lists
                self.backbone_features[name] = (
                    [o.detach() for o in output] if isinstance(output, (list, tuple)) else output
                )
        return hook

    def _detect_input_hook(self, module, input, output):
        # Detect.forward receives a single argument: a list of feature maps [P3, P4, P5, ...]
        x = input[0] if len(input) else None
        if isinstance(x, (list, tuple)):
            self.detect_inputs["features"] = [t.detach() if isinstance(t, torch.Tensor) else t for t in x]
        else:
            self.detect_inputs["features"] = x  # fallback

        return output

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # Reset storages each call
        self.backbone_features = {}
        self.detect_inputs = {}

        _ = self.core(x)  # raw forward through the Model (not the Predictor wrapper)
        detect_feats = self.detect_inputs.get("features", None)
        return self.backbone_features, detect_feats

    def describe_layers(self) -> List[str]:
        """
        Returns a simple map (index -> module class name) to help you pick style_layers.
        """
        return [f"{i}: {m.__class__.__name__}" for i, m in enumerate(self.layers)]
