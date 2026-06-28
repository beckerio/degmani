from pathlib import Path
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from degmani.degmani_utils.arniqua_distortion import distort_images
from degmani.degmani_utils.img_utils import letterbox_tensor, resize_crop_cv

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}"


class ARNIQUA_Base_Dataset(Dataset):
    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        samples_per_dataset: list[int] | int = 1000,
        hard_neg: bool = True,
        channels: int = 3,
        pristine_prob: float = 0.05,
    ):
        super().__init__()
        self.prefix = 'ARNIQUA_Base_Dataset: '
        self.img_paths = img_path if isinstance(img_path, list) else [img_path]
        self.imgsz = imgsz
        self.channels = channels
        self.hard_neg = hard_neg
        self.pristine_prob = pristine_prob
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.samples_per_dataset = self._normalize_samples_per_dataset(samples_per_dataset)

        self.im_files = self.get_img_files(self.img_paths)
        self.ni = len(self.im_files)
        self.ims = [None] * self.ni
        self.im_hw0 = [None] * self.ni
        self.im_hw = [None] * self.ni

    def _normalize_samples_per_dataset(self, samples_per_dataset: list[int] | int) -> list[int]:
        if isinstance(samples_per_dataset, int):
            return [samples_per_dataset] * len(self.img_paths)
        if len(samples_per_dataset) != len(self.img_paths):
            raise ValueError(
                f'{self.prefix}samples_per_dataset length ({len(samples_per_dataset)}) must match '
                f'number of image paths ({len(self.img_paths)}).'
            )
        return list(samples_per_dataset)

    def get_img_files(self, img_paths: list[str]) -> list[str]:
        files = []
        for index, img_path in enumerate(img_paths):
            print(f'{self.prefix}get_img_files :: reading files from {img_path}')
            path = Path(img_path)
            if path.is_dir():
                current_files = glob.glob(str(path / '**' / '*.*'), recursive=True)
                if not current_files:
                    raise FileNotFoundError(f'{self.prefix}No files found under directory {path}')

                sample_count = self.samples_per_dataset[index]
                if sample_count is not None:
                    if sample_count < 0:
                        raise ValueError(f'{self.prefix}sample count must be non-negative, got {sample_count}')
                    current_files = random.choices(current_files, k=sample_count)
                files.extend(current_files)
            elif path.is_file():
                with open(path, encoding='utf-8') as handle:
                    rows = handle.read().strip().splitlines()
                parent = str(path.parent) + os.sep
                files.extend([row.replace('./', parent) if row.startswith('./') else row for row in rows])
            else:
                raise FileNotFoundError(f'{self.prefix}{path} does not exist')

        im_files = sorted(
            file_path.replace('/', os.sep)
            for file_path in files
            if file_path.rpartition('.')[-1].lower() in IMG_FORMATS
        )
        if not im_files:
            raise FileNotFoundError(f'{self.prefix}No images found in {img_paths}. {FORMATS_HELP_MSG}')

        print(f'{self.prefix}get_img_files :: #img {len(im_files)}')
        return im_files

    @staticmethod
    def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
        file_bytes = np.fromfile(filename, np.uint8)
        if filename.endswith((".tiff", ".tif")):
            success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
            if not success:
                return None
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)

        image = cv2.imdecode(file_bytes, flags)
        return image[..., None] if image is not None and image.ndim == 2 else image

    def load_image(self, index: int, rect_mode: bool = True):
        image, image_path = self.ims[index], self.im_files[index]
        if image is None:
            image = self.imread(image_path, flags=self.cv2_flag)
            if image is None:
                raise FileNotFoundError(f'Image not found {image_path}')

            h0, w0 = image.shape[:2]
            if rect_mode:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    width = min(math.ceil(w0 * ratio), self.imgsz)
                    height = min(math.ceil(h0 * ratio), self.imgsz)
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            self.ims[index] = image
            self.im_hw0[index] = (h0, w0)
            self.im_hw[index] = image.shape[:2]
        return self.ims[index], self.im_hw0[index], self.im_hw[index]

    def __len__(self):
        return self.ni

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img_uint8).permute(2, 0, 1).contiguous().float() / 255.0

    def _sample_reference_index(self, index: int) -> int:
        if self.ni <= 1:
            return index
        ref_index = random.randrange(self.ni)
        return ref_index if ref_index != index else (ref_index + 1) % self.ni

    def _distort_pair(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor):
        random_value = random.random()
        if random_value > self.pristine_prob:
            tensor_a_dist, distort_functions, distort_values = distort_images(
                tensor_a,
                max_distortions=5,
                num_levels=5,
            )
            tensor_b_dist, _, _ = distort_images(
                tensor_b,
                distort_functions=distort_functions,
                distort_values=distort_values,
            )
            return tensor_a_dist, tensor_b_dist, distort_functions, distort_values, random_value

        return tensor_a, tensor_b, None, None, random_value

    def __getitem__(self, index):
        img_a, _, _ = self.load_image(index)
        img_b, _, _ = self.load_image(self._sample_reference_index(index))

        img_a = self._to_rgb(img_a)
        img_b = self._to_rgb(img_b)

        tensor_a = self._to_tensor(img_a.copy())
        tensor_b = self._to_tensor(img_b.copy())
        tensor_a_dist, tensor_b_dist, distort_functions, distort_values, random_value = self._distort_pair(tensor_a, tensor_b)

        tensor_a_dist, *_ = letterbox_tensor(tensor_a_dist)
        tensor_b_dist, *_ = letterbox_tensor(tensor_b_dist)
        tensor_a, *_ = letterbox_tensor(tensor_a)
        tensor_b, *_ = letterbox_tensor(tensor_b)

        if not self.hard_neg:
            return tensor_a, tensor_a_dist, tensor_b, tensor_b_dist

        tensor_a_ds = self._to_tensor(resize_crop_cv(img_a).copy())
        tensor_b_ds = self._to_tensor(resize_crop_cv(img_b).copy())
        if random_value > self.pristine_prob:
            tensor_a_dist_ds, _, _ = distort_images(
                tensor_a_ds,
                distort_functions=distort_functions,
                distort_values=distort_values,
            )
            tensor_b_dist_ds, _, _ = distort_images(
                tensor_b_ds,
                distort_functions=distort_functions,
                distort_values=distort_values,
            )
        else:
            tensor_a_dist_ds = tensor_a_ds
            tensor_b_dist_ds = tensor_b_ds

        tensor_a_dist_ds, *_ = letterbox_tensor(tensor_a_dist_ds)
        tensor_b_dist_ds, *_ = letterbox_tensor(tensor_b_dist_ds)
        return tensor_a, tensor_a_dist, tensor_b, tensor_b_dist, tensor_a_dist_ds, tensor_b_dist_ds

    @staticmethod
    def collate_fn(batch):
        if not batch:
            raise ValueError('Empty batch received.')

        sample_len = len(batch[0])
        field_names_by_len = {
            4: ('imgs_a', 'imgs_a_dist', 'imgs_b', 'imgs_b_dist'),
            6: ('imgs_a', 'imgs_a_dist', 'imgs_b', 'imgs_b_dist', 'imgs_a_dist_ds', 'imgs_b_dist_ds'),
        }
        if sample_len not in field_names_by_len:
            raise ValueError(f'Unexpected sample length: {sample_len}')

        field_names = field_names_by_len[sample_len]
        stacked = zip(*batch)
        return {
            field_name: torch.stack(tensors, 0)
            for field_name, tensors in zip(field_names, stacked)
        }
