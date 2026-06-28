import numpy as np
import cv2
import os
import torch

def letterbox_ultra(image, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
    """Return images with added border."""
    img = image
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    return img
    """ end of function ---------------------------------------------------------------------------------------------"""

def padded_resize(input, size):
    c, h, w = input.shape
    input_ = torch.full([3, size, size], 0.5)
    scale = max(w, h) / size
    offset = int((size - w / scale) // 2), int((size - h / scale) // 2)
    input_[
        :,
        offset[1]:offset[1]+int(h / scale),
        offset[0]:offset[0]+int(w / scale),
    ] = torch.nn.functional.interpolate(
        input.unsqueeze(0),
        (int(h / scale), int(w / scale)),
        mode='bilinear')[0]
    return input_