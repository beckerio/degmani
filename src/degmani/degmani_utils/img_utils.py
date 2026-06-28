import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

import torch
import torchvision
import torch.nn.functional as F

from typing import Optional

def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/patches.py
    Read an image from a file with multilanguage filename support.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Controls how the image is read.

    Returns:
        (np.ndarray | None): The read image array, or None if reading fails.

    Examples:
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im  # Always ensure 3 dimensions

def load_image(filename: str, imgsz: int =640, cv2_flag:int =cv2.IMREAD_COLOR, rect_mode: bool = True):
    #cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

    im = imread(filename, flags=cv2_flag)  # BGR
    if im is None:
        raise FileNotFoundError(f"Image Not Found {filename}")

    h0, w0 = im.shape[:2]  # orig hw
    if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
        r = imgsz / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    elif not (h0 == w0 == imgsz):  # resize by stretching image to square imgsz
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    if im.ndim == 2:
        im = im[..., None]
    return im, (h0, w0), im.shape[:2]


def letterbox_tensor(x: torch.Tensor, new_shape=(640, 640), pad_value=114):
    # x: [C,H,W], float in [0,1] or integer [0..255]
    assert x.ndim == 3, "letterbox_tensor expects [C,H,W]"
    C, H, W = x.shape
    nh, nw = new_shape
    r = min(nh / H, nw / W)
    new_h, new_w = int(math.floor(H * r)), int(math.floor(W * r))

    # resize
    mode = 'bilinear' if x.is_floating_point() else 'nearest'
    x = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode=mode,
                      align_corners=False if mode == 'bilinear' else None).squeeze(0)

    # padding
    pad_h, pad_w = nh - new_h, nw - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    pad_val = (pad_value / 255.0) if x.is_floating_point() else int(pad_value)

    x = F.pad(x, (left, right, top, bottom), value=pad_val)
    return x, r, (left, top)


def resize_crop_cv(img: np.ndarray, crop_size: Optional[int] = 220, downscale_factor: int = 2, center: bool = True) -> np.ndarray:
    """
    Resize by downscale_factor, then crop. No zero-padding:
    if the image is smaller than crop_size, the crop shrinks to fit.
    Returns the crop as-is (variable size).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array (OpenCV image).")
    if downscale_factor < 1:
        raise ValueError("downscale_factor must be >= 1.")

    h, w = img.shape[:2]

    # Downscale
    if downscale_factor > 1:
        new_w = max(1, w // downscale_factor)
        new_h = max(1, h // downscale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if crop_size is None:
        return img

    # Adapt crop size to image dimensions
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)

    if center:
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    else:
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

    crop = img[top:top + crop_h, left:left + crop_w]
    img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def hard_negative_downsample_upsample(img_uint8: np.ndarray, ds_factor: int = 2):
    # Step 1: Downsample by a factor of ds_factor
    h, w = img_uint8.shape[:2]  # orig hw

    img_down = cv2.resize(img_uint8, (w // ds_factor, h // ds_factor), interpolation=cv2.INTER_LINEAR)
    # Step 2: Upsample by a factor of ds_factor
    h_ds, w_ds = img_down.shape[:2]  # orig hw
    img_up = cv2.resize(img_uint8, (w_ds * ds_factor, h_ds * ds_factor), interpolation=cv2.INTER_LINEAR)
    return img_up


def resize_crop_cv_border(img: np.ndarray, crop_size: Optional[int] = 220, downscale_factor: int = 2,
                          center: bool = True) -> np.ndarray:
    """
    Resize the image with the desired downscale factor and optionally crop it to the desired size.
    The crop is randomly sampled from the image. If crop_size is None, no crop is applied.
    If the crop goes out of bounds, the result is padded with zeros (black).

    Args:
        img (np.ndarray): OpenCV image (H x W x C or H x W), dtype preserved.
        crop_size (int | None): size of the square crop. If None, no crop is applied.
        downscale_factor (int): integer factor to downscale the image by.

    Returns:
        np.ndarray: resized and/or cropped image.
        :param center:
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array (OpenCV image).")
    if downscale_factor < 1:
        raise ValueError("downscale_factor must be >= 1.")

    h, w = img.shape[:2]

    # Downscale
    if downscale_factor > 1:
        new_w = max(1, w // downscale_factor)
        new_h = max(1, h // downscale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Random crop with zero padding if needed
    if crop_size is not None:
        if center:

            # Prepare output crop filled with zeros (black)
            if img.ndim == 2:
                crop = np.zeros((crop_size, crop_size), dtype=img.dtype)
            else:
                crop = np.zeros((crop_size, crop_size, img.shape[2]), dtype=img.dtype)

            # Compute source and destination ranges to center the image in the crop
            y_len = min(crop_size, h)
            x_len = min(crop_size, w)

            y_src0 = (h - y_len) // 2
            x_src0 = (w - x_len) // 2

            y_dst0 = (crop_size - y_len) // 2
            x_dst0 = (crop_size - x_len) // 2

            crop[y_dst0:y_dst0 + y_len, x_dst0:x_dst0 + x_len] = img[y_src0:y_src0 + y_len, x_src0:x_src0 + x_len]
            img = crop
        else:
            # Choose top-left so that if image is smaller than crop, we start at (0,0)
            max_top = max(1, h - crop_size + 1)  # +1 to make the last valid index inclusive
            max_left = max(1, w - crop_size + 1)
            top = np.random.randint(0, max_top)
            left = np.random.randint(0, max_left)

            # Prepare output crop filled with zeros
            if img.ndim == 2:
                crop = np.zeros((crop_size, crop_size), dtype=img.dtype)
            else:
                channels = img.shape[2]
                crop = np.zeros((crop_size, crop_size, channels), dtype=img.dtype)

            # Compute overlapping region between source and desired crop
            y1 = min(h, top + crop_size)
            x1 = min(w, left + crop_size)

            # Copy the overlapping region into the zero-padded crop
            crop[0:y1 - top, 0:x1 - left] = img[top:y1, left:x1]
            img = crop
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def to_imshow_array(img: torch.Tensor):
    """
    Convert a torch image tensor to a numpy array suitable for plt.imshow.
    Accepts:

      - (3,H,W) or (1,H,W) or (H,W,3) or (H,W) or (1,3,H,W)

    Returns:

      - (H,W,3) float32 in [0,1] or (H,W) for grayscale

    """
    if torch.is_tensor(img):
        x = img
        # remove batch if present
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).contiguous()
        # (H,W,1) -> (H,W)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # ensure float in [0,1]
        if x.dtype.is_floating_point:
            x = x.clamp(0, 1)
        else:
            # uint8 -> float [0,1]
            x = x.to(torch.float32) / 255.0
        return x.detach().cpu().numpy()
    else:
        # numpy path (assume already HWC or HW)
        return img

def preprocess_for_yolo(
    img_path: Path,
    imgsz: int,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Load image, resize/letterbox, and return:
      - orig_rgb: original RGB image as numpy array for visualization
      - tensor: [1,3,H,W] float tensor on device in [0,1] (BGR for model)

    """
    bgr = imread(str(img_path), flags=cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image Not Found {img_path}")

    h0, w0 = bgr.shape[:2]
    scale = imgsz / max(h0, w0)
    if scale != 1:
        new_w = min(int(np.ceil(w0 * scale)), imgsz)
        new_h = min(int(np.ceil(h0 * scale)), imgsz)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(bgr).permute(2, 0, 1).float().div_(255.0)
    t, *_ = letterbox_tensor(t)  # [3, H, W]
    t = t.unsqueeze(0).to(device)  # [1,3,H,W]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, t