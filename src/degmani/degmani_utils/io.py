import numpy as np
import cv2
import os
import glob
import re
import pickle as pkl
import datetime
import json
import yaml
from pathlib import Path
from typing import List, Optional, Sequence
import random

img_formats = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng')

def load_img_from_folder_sampled(
    path: str,
    sample_count: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    randomize: bool = True,      # default: random
    replace: bool = False,       # True -> use random.choices (duplicates allowed)
    seed: Optional[int] = None,
    img_formats: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,  # optional sampling weights
) -> np.ndarray:
    """
    Load image file paths and sample by fixed count or fraction.

    Args:
        path: Root directory to walk.
        sample_count: Return up to this many samples.
        sample_fraction: Return ceil(fraction * total) samples (0 < fraction <= 1).
        randomize: If True (default), sample randomly; else take first k in traversal order.
        replace: If True, sample with replacement (duplicates allowed) using random.choices.
                 If False, sample without replacement.
        seed: Random seed for reproducibility.
        img_formats: Extensions/substrings to match. Defaults to common image types or global `img_formats`.
        weights: Optional weights/probabilities for sampling. Works with both replace=False (numpy) and
                 replace=True (random.choices). Length must match the number of files.

    Returns:
        np.ndarray of selected file paths.
    """
    if img_formats is None:
        img_formats = globals().get("img_formats") or (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            fname = file.lower()
            if any(strtocheck in fname for strtocheck in img_formats):
                file_list.append(os.path.join(root, file))

    n = len(file_list)
    if n == 0:
        return np.array([], dtype=object)

    if sample_count is not None and sample_fraction is not None:
        raise ValueError("Provide either `sample_count` or `sample_fraction`, not both.")

    if sample_count is None and sample_fraction is None:
        return np.array(file_list, dtype=object)

    # Determine number to sample
    if sample_count is not None:
        if sample_count <= 0:
            raise ValueError("`sample_count` must be > 0.")
        k = min(sample_count, n) if not replace else sample_count
    else:
        f = float(sample_fraction)
        if not (0 < f <= 1):
            raise ValueError("`sample_fraction` must be in (0, 1].")
        k = max(1, int(np.ceil(f * n)))

    if not randomize:
        selected = file_list[:k]
    else:
        if replace:
            if seed is not None:
                random.seed(seed)
            if weights is not None:
                if len(weights) != n:
                    raise ValueError("`weights` length must match number of files.")
                selected = random.choices(file_list, weights=weights, k=k)
            else:
                selected = random.choices(file_list, k=k)
        else:
            rng = np.random.default_rng(seed)
            p = None
            if weights is not None:
                if len(weights) != n:
                    raise ValueError("`weights` length must match number of files.")
                total = float(sum(weights))
                if total <= 0:
                    raise ValueError("`weights` must sum to > 0.")
                p = np.asarray(weights, dtype=float) / total
            idx = rng.choice(n, size=k, replace=False, p=p)
            selected = [file_list[i] for i in idx]

    return np.array(selected, dtype=object)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def gen_time_stamp():
    """ ---------------------------------------------------------------------------
    Generate time stamp string

    Returns:
      String
    ------------------------------------------------------------------------------- """
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    """ End of function ----------------------------------------------------------- """


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    """ end of function  ----------------------------------------------------------- """


def load_files_folder(
        path: str,
        filetype: str = ".txt",
) -> np.ndarray:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                file_list.append(os.path.join(root, file))
    return np.array(file_list)
    """ end of function ---------------------------------------------------------------------------------------------"""

def load_files_folder_list(
        path: str,
        filetype: str = ".txt",
) -> np.ndarray:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                file_list.append(os.path.join(root, file))
    return file_list
    """ end of function ---------------------------------------------------------------------------------------------"""

def load_img_from_folder(
        path: str,
) -> np.ndarray:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(strtocheck in file.lower() for strtocheck in img_formats):
                file_list.append(os.path.join(root, file))
    return np.array(file_list)
    """ end of function ---------------------------------------------------------------------------------------------"""

def load_img_from_folder_list(
        path: str,
) -> List:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(strtocheck in file.lower() for strtocheck in img_formats):
                file_list.append(os.path.join(root, file))
    return file_list
    """ end of function ---------------------------------------------------------------------------------------------"""

def file_to_img(filepath: str):
    img = cv2.imread(filepath)  # BGR
    return img
    """ end of function ---------------------------------------------------------------------------------------------"""

def load_files_folder_recursive(path, filetype=".xml"):

    file_list = []
    def generate_all_files(root: Path, only_files: bool = True):
        for p in root.rglob("*"):
            if only_files and not p.is_file():
                continue
            yield p

    for p in generate_all_files(Path(path), only_files=False):
        if p.suffix == (filetype):
            file_list.append(str(p))
            #file_list.append(os.path.join(root, file))

    return np.array(file_list)
    """ end of function ---------------------------------------------------------------------------------------------"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
def load_json(load_path: str, encoding: str = "utf-8"):
    """
    Loads json formatted data (given as "data") from load_path
    Encoding type can be specified with 'encoding' argument
    Example inputs:
        load_path: "dirname/coco.json"
    """
    # read from path
    with open(load_path, encoding=encoding) as json_file:
        data = json.load(json_file)
    return data

def save_json(data, save_path):
    """
    Saves json formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.json"
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), cls=NumpyEncoder)


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data