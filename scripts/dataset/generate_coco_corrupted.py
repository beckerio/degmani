import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from degmani.degmani_utils.image_corruptions import corruption_dict, corruption_tuple
from degmani.degmani_utils.io import file_to_img, load_img_from_folder_list


DEFAULT_CORRUPTION_LEVELS = [1, 2, 3, 4, 5]
DEFAULT_DATASET_DIR = "/data/SSD/datasets/COCO/data/val2017"


def corrupt(image_rgb, severity=1, corruption_name=None, corruption_number=None):
    if corruption_number is not None:
        corruption_fn = corruption_tuple[corruption_number]
    elif corruption_name is not None:
        corruption_fn = corruption_dict[corruption_name]
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed.")

    corrupted = corruption_fn(Image.fromarray(image_rgb), severity)
    if corrupted.shape != image_rgb.shape:
        raise AssertionError("Output image is not the same size as input image.")

    return np.uint8(corrupted)


def collect_image_paths(dataset_dirs, image_subfolder):
    img_files = []
    for dataset_dir in dataset_dirs:
        image_dir = Path(dataset_dir) / image_subfolder
        print(f"loading images from {image_dir}")
        img_files.extend(load_img_from_folder_list(image_dir))

    return sorted(img_files)


def output_dir_for(dataset_dir, corruption_level):
    output_dir = Path(dataset_dir).expanduser() / f"images_c{corruption_level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_rgb_image(img_path):
    img_bgr = file_to_img(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"Could not write image: {path}")


def prepare_corrupted_dataset(opt):
    print("prepare_corrupted_dataset :: start")

    img_files = collect_image_paths(opt.dataset_dir, opt.img_subfolder_name)
    print(f"dataset directories: {opt.dataset_dir}")
    print(f"all imgs: {len(img_files)}")

    if not img_files:
        raise ValueError("No images found.")

    for corruption_level in opt.corruption_levels:
        print(f"Generating corrupted data at corruption level {corruption_level}")

        for img_path in tqdm(img_files, total=len(img_files)):
            path_helper = Path(img_path)
            image_rgb = load_rgb_image(img_path)

            corrupt_img = corrupt(
                image_rgb,
                severity=corruption_level,
                corruption_name=opt.corruption_name,
                corruption_number=opt.corruption_number,
            )

            if opt.save_img:
                output_dir = output_dir_for(path_helper.parent.parent, corruption_level)
                save_rgb_image(output_dir / path_helper.name, corrupt_img)

            if opt.show_img:
                cv2.imshow("original", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow(f"{opt.corruption_name} severity {corruption_level}", cv2.cvtColor(corrupt_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("prepare_corrupted_dataset :: end")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        nargs="+",
        default=[DEFAULT_DATASET_DIR],
        help="Parent dataset directory containing the image subfolder.",
    )
    parser.add_argument("--img_subfolder_name", type=str, default="images", help="Image subfolder name.")
    parser.add_argument(
        "--corruption_levels",
        nargs="+",
        type=int,
        default=DEFAULT_CORRUPTION_LEVELS,
        help="Corruption severity levels to generate.",
    )
    parser.add_argument(
        "--corruption_name",
        type=str,
        default="gaussian_noise",
        choices=sorted(corruption_dict),
        help="Corruption function to apply.",
    )
    parser.add_argument(
        "--corruption_number",
        type=int,
        default=None,
        choices=range(len(corruption_tuple)),
        help="Corruption index to apply instead of corruption_name.",
    )
    parser.add_argument("--show_img", action=argparse.BooleanOptionalAction, default=False, help="Show generated images.")
    parser.add_argument("--save_img", action=argparse.BooleanOptionalAction, default=True, help="Save generated images.")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    opt = parse_args()
    print("***********************************************************************************************************")
    print("Options:")
    for arg in vars(opt):
        print(f"{arg}: {getattr(opt, arg)}")
    print("***********************************************************************************************************")

    prepare_corrupted_dataset(opt)
