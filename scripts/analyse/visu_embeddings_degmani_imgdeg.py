import argparse
import math
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm import tqdm

from degmani.config.checkpoints_folder import CHECKPOINT_ROOT, checkpoint_path
from degmani.degmani_utils.arniqua_distortion import PRISTINE_GROUP, PRISTINE_KEY
from degmani.degmani_utils.eval_utils import compute_overall_mean
from degmani.degmani_utils.img_utils import letterbox_tensor
from degmani.degmani_utils.fm_extractor import capture_by_indices
from degmani.degmani_utils.io import file_to_img, load_img_from_folder_list, yaml_load
from degmani.models.degmani import Degradation_Manifold

"""
https://github.com/beckerio/imdeg
"""
from imdeg import apply_degradation, list_paper_types, map_paper_term
from imdeg.config.papers import PAPER_NAME_MAP, PaperSelection

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

EPS = 1e-9


def area_under_gaussian_at_left(t,mu,sigma):
    a = t-mu
    b = math.sqrt(2)*sigma
    return .5*(1+scipy.special.erf(a/b))

def area_under_gaussian_at_right(t,mu,sigma):
    return 1-area_under_gaussian_at_left(t,mu,sigma)

my_threshold = 6
use_react = True


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_uint8).permute(2, 0, 1).contiguous().float() / 255.0

def get_img_files(image_lst: List[str] | str, samples_subfolder: int) -> List[str]:
    img_sources = image_lst if isinstance(image_lst, list) else [image_lst]
    img_files: List[str] = []
    for img_source in img_sources:
        print(f"loading images from {img_source}")
        available_files = load_img_from_folder_list(img_source)
        if not available_files:
            raise FileNotFoundError(f"No images found in {img_source}")
        img_files.extend(random.choices(available_files, k=samples_subfolder))
    return img_files


def calc_cluster(tsne, mylabels):
    ## calculate cluster mean
    clusters = tsne.reshape(len(mylabels), -1, 2)

    # Step 1: Compute the mean for each cluster
    cluster_means = clusters.mean(axis=1)  # Shape (X, 2)
    cluster_stds = clusters.std(axis=1)  # Shape (X, 2)

    # Step 2: Compute mean distance from each cluster to the rest
    mean_distances = []
    for i in range(len(mylabels)):
        rest_means = np.delete(cluster_means, i, axis=0)  # Remove the current cluster mean
        mean_of_rest = rest_means.mean(axis=0)  # Mean of the rest clusters

        distance = np.linalg.norm(cluster_means[i] - mean_of_rest)  # Euclidean distance
        mean_distances.append(distance)

    mean_distances = np.array(mean_distances)

    pairwise_distances = cdist(cluster_means, cluster_means)  #
    mean_distances_to_others = pairwise_distances.mean(axis=1)  # Mean distance for each cluster

    print("Cluster Means:\n", cluster_means)
    print("Mean Distances to the Rest:\n", mean_distances)
    print("Mean Distances to the Rest:\n", mean_distances_to_others)
    print("pairwise_distances:\n", pairwise_distances)

    # Colormap and normalization
    # cmap = matplotlib.colormaps['Set3']#cm.get_cmap('Set3')  # Colormap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(mylabels) + 1)  # Normalization
    cmap = plt.get_cmap('Set3')
    # colors = [cmap(i / len(mylabels)) for i in range(len(mylabels))]
    # Generate consistent colors for clusters
    colors = [cmap(norm(i)) for i in range(len(mylabels))]

    fig, ax = plt.subplots()
    # Plot cluster means, clusters, and ellipses
    for i, (mean, std, color) in enumerate(zip(cluster_means, cluster_stds, colors)):
        # Plot cluster mean
        ax.scatter(mean[0], mean[1], color=color, s=100, edgecolor='k')
        # Plot individual points in the cluster
        ax.scatter(clusters[i, :, 0], clusters[i, :, 1], s=40, color=color, edgecolors='black')
        # Plot ellipse representing the cluster's standard deviation
        ellipse = plt.Circle(mean, std.max(), color=color, alpha=0.3)
        ax.add_patch(ellipse)

    # Create the color bar
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        # color=colors,
        ax=ax,
        boundaries=np.arange(len(mylabels) + 1) - 0.5,
        ticks=np.arange(len(mylabels)),
        orientation='vertical'
    )

    # Add tick labels and color bar title
    colorbar.set_ticklabels(mylabels)
    # colorbar.set_label(mylabels, rotation=270, labelpad=20)

    # Add plot details
    ax.set_title('t-SNE dataset embeddings')
    ax.set_xlabel('t-SNE $x$')
    ax.set_ylabel('t-SNE $y$')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))  # Adjust legend position

    # Show the plot
    plt.tight_layout()
    # save_img_name = os.path.join(opt.out_folder, "dataset_embeddings_tsne_pbvs_all.jpg")
    # plt.savefig(save_img_name, dpi=300)
    plt.show()



agnolucci_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    #"jpeg": ["jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness change": ["brighten", "darken", "meanshift"],
    "spatial distortion": ["jitter", "noneccpatch", "pixelate", "quantization", "colorblock"],
    "sharpness contrast": ["highsharpen", "lincontrchange", "nonlincontrchange"],
}

full_names_agnolucci = {
    "brighten": "Brighten",
    "darken": "Darken",
    "meanshift": "Mean Shift",

    "gaublur": "Gaussian Blur",
    "lensblur": "Lens Blur",
    "motionblur": "Motion Blur",

    "colordiff": "Color Diffusion",
    "colorshift": "Color Shift",
    "colorsat1": "Color Saturation 1",
    "colorsat2": "Color Saturation 2",

    "jpeg2000": "JPEG2000",
    "jpeg": "JPEG",

    "whitenoise": "White Noise",
    "whitenoiseCC": "White Noise CC",# "White noise cc in color component"
    "impulsenoise": "Impulse Noise",
    "multnoise": "Multiplicative Noise",

    "jitter": "Jitter",
    "noneccpatch": "Non-Eccentricity Patch",
    "pixelate": "Pixelate",
    "quantization": "Quantization",
    "colorblock": "Color Block",

    "highsharpen": "High Sharpen",
    "lincontrchange": "Linear Contrast Change",
    "nonlincontrchange": "Nonlinear Contrast Change",

}


hendrycks_groups = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"]
}

full_names_hendrycks = {
    "gaussian_noise": "Gaussian Noise",
    "shot_noise": "Shot Noise",
    "impulse_noise": "Impulse Noise",
    "defocus_blur": "Defocus Blur",
    "gaussian_blur": "Gaussian Blur",
    "speckle_noise": "Speckle Noise",
    "glass_blur": "Glass Blur",
    "motion_blur": "Motion Blur",
    "zoom_blur": "Zoom Blur",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Brightness",
    "spatter": "Spatter",
    "contrast": "Contrast",
    "saturate": "Saturate",
    "elastic_transform": "Elastic Transform",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG Compression"
}

hendrycks_group_palettes = {
    "noise": ["#c6dbef", "#6baed6", "#2171b5", "blue"],                   # Blues
    "blur": ["#d4b9da", "#9e9ac8", "#6a51a3", "#3f007d", "purple"],         # Purples
    "weather": ["#fee391", "#fec44f", "#fe9929", "#ec7014", "yellow"],      # Yellows/Oranges
    "digital": ["#fbb4ae", "#f768a1", "#ae017e", "#49006a", "pink"]       # Pinks/Violets
}


Gold = ["#fff3b0", "#f9c74f", "#c28f2c"]
agnolucci_group_palettes = {
    "brightness change": Gold, #["#c6dbef", "#6baed6", "#2171b5"],                           #
    "blur": ["#d4b9da", "#9e9ac8", "#6a51a3"],                                        # Purples
    "color distortion": ["#c7e9c0", "#74c476", "#31a354", "#006d2c"],                  # Greens
    "jpeg": ["#bdbdbd", "#636363"],                                                   # Greys
    "noise": ["#fdd0a2", "#fdae6b", "#fd8d3c", "#d94801"],                             # Oranges
    "spatial distortion": ["#fde0dd", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e"],     # RdPu
    "sharpness contrast": ["#fcbba1", "#fb6a4a", "#cb181d"],                           # Reds
}


GROUP_MARKERS = {
    "brightness change": "o",   # circle
    "blur": "s",                # square
    "color distortion": "^",    # triangle up
    "jpeg": "D",                # diamond
    "noise": "P",               # plus (filled)
    "spatial distortion": "X",  # X (filled)
    "sharpness contrast": "v",  # triangle down
    "weather": "H", # hexagone
    # Optional if you add pristine as a "group"
    "Pristine": "*",
}

MARKER_UNICODE = {
    'o': '●',     # circle
    's': '■',     # square
    '^': '▲',     # triangle up
    'v': '▼',     # triangle down
    '<': '◀',     # triangle left
    '>': '▶',     # triangle right
    'D': '◆',     # diamond
    'd': '◆',     # thin diamond approx
    'P': '✚',     # filled plus (approx)
    'X': '✖',     # filled x (approx)
    '*': '★',     # star
    'h': '⬣',     # hexagon1 approx
    'H': '⬢',     # hexagon2 approx
    'p': '⬟',     # pentagon approx
    '8': '❂',     # octagon approx
    None: '●'
}

def _display_group_name(g):
    # Normalize then format for display
    g_norm = _norm_group_name(g)
    mapping = {
        'noise': 'Noise',
        'blur': 'Blur',
        'weather': 'Weather',
        'digital': 'Digital',
        'pristine': 'Pristine',
        'unknown': 'Unknown',
    }
    return mapping.get(g_norm, g_norm.capitalize())

def get_nice_mapping_hendrycks() -> Dict[str, Tuple[str, str]]:
    nice_mapping = {}
    for group, keys in hendrycks_groups.items():
        palette = hendrycks_group_palettes[group]
        if len(palette) != len(keys):
            raise ValueError(f"Palette mismatch in group '{group}'")
        for key, color in zip(keys, palette):
            nice_mapping[key] = (full_names_hendrycks[key], color)

    # Optional: add pristine entry
    nice_mapping["Pristine"] = ("Pristine", "#17BECF")
    return nice_mapping

def get_nice_mapping_agnolucci() -> Dict[str, Tuple[str, str]]:
    nice_mapping = {}
    for group, keys in agnolucci_groups.items():
        palette = agnolucci_group_palettes[group]
        if len(palette) != len(keys):
            raise ValueError(f"Palette mismatch in group '{group}'")
        for key, color in zip(keys, palette):
            nice_mapping[key] = (full_names_agnolucci[key], color)

    # Optional: add pristine entry
    nice_mapping["Pristine"] = ("Pristine", "#17BECF")
    return nice_mapping


def _format_term_name(term: str) -> str:
    return str(term).replace("_", " ").replace("-", " ").strip().title()


def build_plot_config(paper_key: str, paper_types: List[Dict[str, str]]):
    paper_key = paper_key.lower()
    if paper_key == "hendrycks":
        return get_nice_mapping_hendrycks(), hendrycks_groups, GROUP_MARKERS
    if paper_key == "agnolucci":
        return get_nice_mapping_agnolucci(), agnolucci_groups, GROUP_MARKERS

    generic_groups: Dict[str, List[str]] = {}
    for row in paper_types:
        term = row["term"]
        group_name = row.get("group_name", "unknown")
        generic_groups.setdefault(group_name, []).append(term)

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "H", "<", ">", "d", "p", "8"]
    generic_markers = {
        group_name: marker_cycle[index % len(marker_cycle)]
        for index, group_name in enumerate(generic_groups)
    }
    generic_markers[PRISTINE_GROUP] = generic_markers.get(PRISTINE_GROUP, "*")

    total_terms = sum(len(terms) for terms in generic_groups.values())
    palette = cm.get_cmap("tab20", max(total_terms, 1))
    nice_mapping: Dict[str, Tuple[str, str]] = {}
    color_index = 0
    for group_name, terms in generic_groups.items():
        for term in terms:
            nice_mapping[term] = (_format_term_name(term), matplotlib.colors.to_hex(palette(color_index)))
            color_index += 1
    nice_mapping[PRISTINE_KEY] = ("Pristine", "#17BECF")
    return nice_mapping, generic_groups, generic_markers

from matplotlib.colors import ListedColormap, BoundaryNorm

def encode_labels_with_colors(distortion_keys, nice_map):
    name_to_id, class_names, class_colors, class_keys, labels_idx = {}, [], [], [], []
    for key in distortion_keys:
        name, color = nice_map[key]
        if name not in name_to_id:
            name_to_id[name] = len(class_names)
            class_names.append(name)
            class_colors.append(color)
            class_keys.append(key)
        labels_idx.append(name_to_id[name])
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, len(class_names))
    return np.array(labels_idx), class_names, class_keys, cmap, norm

def _norm_group_name(g):
    return str(g).strip().lower() if g is not None else "unknown"

def _build_dist_to_group_map(hendrycks_groups, PRISTINE_KEY, PRISTINE_GROUP):
    # Normalize group keys to lowercase
    dist_to_group = {}
    for g, ds in hendrycks_groups.items():
        g_norm = _norm_group_name(g)
        for d in ds:
            dist_to_group[d] = g_norm
    dist_to_group[PRISTINE_KEY] = _norm_group_name(PRISTINE_GROUP)
    return dist_to_group

def _canonical_markers(GROUP_MARKERS, PRISTINE_GROUP):
    # Normalize marker keys to lowercase
    canon = { _norm_group_name(k): v for k, v in GROUP_MARKERS.items() }
    # Ensure pristine exists
    canon.setdefault(_norm_group_name(PRISTINE_GROUP), canon.get("pristine", "X"))
    return canon

def plot_tsne_ba(tsne, all_distortion_keys, nice_map, num_levels, out_path,
              hendrycks_groups, GROUP_MARKERS, MARKER_UNICODE, PRISTINE_KEY, PRISTINE_GROUP,
              show=True):
    # Labels -> integers, names, colormap
    labels_idx, class_names, class_keys, cmap, norm = encode_labels_with_colors(all_distortion_keys, nice_map)

    # Build normalized maps
    dist_to_group = _build_dist_to_group_map(hendrycks_groups, PRISTINE_KEY, PRISTINE_GROUP)
    canon_markers = _canonical_markers(GROUP_MARKERS, PRISTINE_GROUP)

    # Resolve group per sample; unknown keys get "unknown"
    group_per_sample = []
    unknown_terms = []
    for k in all_distortion_keys:
        g = dist_to_group.get(k)
        if g is None:
            unknown_terms.append(k)
            group_per_sample.append("unknown")
        else:
            group_per_sample.append(_norm_group_name(g))
    group_per_sample = np.array(group_per_sample)

    # Debug: missing mappings and group counts
    if unknown_terms:
        unk_counts = Counter(unknown_terms)
        print("[plot_tsne] Warning: terms not found in hendrycks_groups -> mapped to 'unknown':")
        for term, cnt in unk_counts.items():
            print(f"  - {term}: {cnt} samples")
    grp_counts = Counter(group_per_sample)
    print("[plot_tsne] Samples per normalized group:")
    for g, c in grp_counts.items():
        print(f"  {g}: {c}")

    # Colorbar labels: prepend marker symbol of the class' group
    cb_labels = []
    for key, name in zip(class_keys, class_names):
        g = dist_to_group.get(key, "unknown")
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        symbol = MARKER_UNICODE.get(marker, "●")
        #name_c = name.capitalize()
        cb_labels.append(f"{symbol} {name}")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot per group with markers; colors from labels_idx via cmap/norm
    # Iterate over groups present in data to avoid missing groups
    present_groups = sorted(set(group_per_sample.tolist()))
    for g in present_groups:
        mask = (group_per_sample == g)
        if not np.any(mask):
            continue
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        #marker = canon_markers.get(g, "o")
        ax.scatter(
            tsne[mask, 0], tsne[mask, 1],
            s=40,
            c=labels_idx[mask],
            cmap=cmap, norm=norm,
            marker=marker,
            edgecolors='black', linewidths=0.4, alpha=0.9,
            label=g
        )

    ax.set_aspect('equal', 'datalim')
    ax.set_title(f'Degradation manifold || Severity level {num_levels}')
    ax.set_xlabel('t-SNE $x$')
    ax.set_ylabel('t-SNE $y$')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # Discrete colorbar for distortions
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    boundaries = np.arange(len(class_names) + 1) - 0.5
    ticks = np.arange(len(class_names))
    cbar = fig.colorbar(sm, ax=ax, boundaries=boundaries, ticks=ticks)
    cbar.set_ticklabels(cb_labels)

    # Legend for groups (marker shapes only)
    handles, labels = [], []
    for g in present_groups:
        #marker = canon_markers.get(g, "o")
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        h = Line2D([], [], marker=marker, linestyle='None',
                   markerfacecolor='white', markeredgecolor='black', markersize=8)
        handles.append(h); labels.append(g.capitalize())
    if handles:
        ax.legend(handles, labels, title="Groups (marker)", loc="upper right", frameon=True,
                  fontsize = 'x-small',
                  title_fontsize='x-small')
        #ax.legend.get_title().set_fontsize('x-small')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #fig.savefig(out_path, dpi=300)
    #fig.savefig(out_path, format="pdf")
    fig.savefig(out_path + ".jpg", dpi=300)
    fig.savefig(out_path + ".pdf", format="pdf")
    if show:
        plt.show()
    plt.close(fig)


def plot_tsne(tsne, all_distortion_keys, nice_map, num_levels, out_path,
              hendrycks_groups, GROUP_MARKERS, MARKER_UNICODE, PRISTINE_KEY, PRISTINE_GROUP,
              show=True):
    # Labels -> integers, names, colormap
    labels_idx, class_names, class_keys, cmap, norm = encode_labels_with_colors(all_distortion_keys, nice_map)

    # Build normalized maps
    dist_to_group = _build_dist_to_group_map(hendrycks_groups, PRISTINE_KEY, PRISTINE_GROUP)
    canon_markers = _canonical_markers(GROUP_MARKERS, PRISTINE_GROUP)

    # Resolve group per sample; unknown keys get "unknown"
    group_per_sample = []
    unknown_terms = []
    for k in all_distortion_keys:
        g = dist_to_group.get(k)
        if g is None:
            unknown_terms.append(k)
            group_per_sample.append("unknown")
        else:
            group_per_sample.append(_norm_group_name(g))
    group_per_sample = np.array(group_per_sample)

    # Debug: missing mappings and group counts
    if unknown_terms:
        unk_counts = Counter(unknown_terms)
        print("[plot_tsne] Warning: terms not found in hendrycks_groups -> mapped to 'unknown':")
        for term, cnt in unk_counts.items():
            print(f"  - {term}: {cnt} samples")
    grp_counts = Counter(group_per_sample)
    print("[plot_tsne] Samples per normalized group:")
    for g, c in grp_counts.items():
        print(f"  {g}: {c}")

    # Colorbar labels: prepend marker symbol of the class' group
    cb_labels = []
    for key, name in zip(class_keys, class_names):
        g = dist_to_group.get(key, "unknown")
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        symbol = MARKER_UNICODE.get(marker, "●")
        cb_labels.append(f"{symbol} {name}")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Determine groups present in data
    present_groups = set(group_per_sample.tolist())

    # ---- Force a specific group to be plotted last ----
    special_last = "pristine"  # change this to the group you want last
    ordered_groups = sorted(present_groups)
    if special_last in ordered_groups:
        ordered_groups = [g for g in ordered_groups if g != special_last] + [special_last]

    # Plot per group with markers; colors from labels_idx via cmap/norm
    for g in ordered_groups:
        mask = (group_per_sample == g)
        if not np.any(mask):
            continue
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        ax.scatter(
            tsne[mask, 0], tsne[mask, 1],
            s=40,
            c=labels_idx[mask],
            cmap=cmap, norm=norm,
            marker=marker,
            edgecolors='black', linewidths=0.4, alpha=0.9,
            label=g
        )

    ax.set_aspect('equal', 'datalim')
    ax.set_title(f'Degradation manifold || Severity level {num_levels}')
    ax.set_xlabel('t-SNE $x$')
    ax.set_ylabel('t-SNE $y$')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # Discrete colorbar for distortions
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    boundaries = np.arange(len(class_names) + 1) - 0.5
    ticks = np.arange(len(class_names))
    cbar = fig.colorbar(sm, ax=ax, boundaries=boundaries, ticks=ticks)
    cbar.set_ticklabels(cb_labels)

    # Legend for groups (marker shapes only), matching the same order
    handles, labels = [], []
    for g in ordered_groups:
        g_norm = _norm_group_name(g)
        marker = canon_markers.get(g_norm, "o")
        h = Line2D([], [], marker=marker, linestyle='None',
                   markerfacecolor='white', markeredgecolor='black', markersize=8)
        handles.append(h)
        labels.append(g.capitalize())

    if handles:
        ax.legend(
            handles, labels,
            title="Groups (marker)",
            loc="upper right",
            frameon=True,
            fontsize='x-small',        # labels
            title_fontsize='x-small'   # title
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".jpg", dpi=300)
    fig.savefig(out_path + ".pdf", format="pdf")
    if show:
        plt.show()
    plt.close(fig)


def run(opt):
    #matplotlib.use('Agg')
    matplotlib.use('TkAgg')

    paper_enum = PaperSelection[opt.paper.upper()]
    paper_name = PAPER_NAME_MAP[paper_enum]

    # metric = "1-ssim"
    print(f"Degradations for paper '{paper_name}' ")
    # print(f"Results (if saved) will go to: {out_dir}")
    #input()
    # get all degradation function from benchmark/paper bachend
    # list_paper_types should return rows with keys: "term", "group_id", "group_name"
    types_arn = list_paper_types(paper_name)

    # with open(out_path, "w") as f, redirect_stdout(f):
    for index, row in enumerate(types_arn):
        print(index + 1, row["term"], "->", row["group_id"], row["group_name"])
    #input()
    nice_map, paper_groups, group_markers = build_plot_config(opt.paper, types_arn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # folder with weights/checkpoints for pretrained detection model (e.g. YOLO) ans distortion models
    checkpt_folder = opt.checkpt_folder


    training_settings = yaml_load(f"{checkpt_folder}/{opt.distmani_folder}/opt.yaml")

    print("Training settings")
    for key, value in training_settings.items():
        print(f"{key}: {value}")

    weights = opt.weights_detector

    #layer_indexes = [10]
    if 'layer_indexes' not in training_settings.keys():
        print("⚠️  old checkpoints -> set layer_indexes manually ⚠️")
        layer_indexes = [0, 1, 3, 5, 7, 10]
    else:
        layer_indexes = training_settings['layer_indexes']

    distmani = Degradation_Manifold(weights=weights,
                                      layer_idxs=layer_indexes,
                                      embedding_dim=training_settings['embedding_dim'],
                                      train_backbone=True,
                                      device=device)

    checkpoint = torch.load(opt.weights_distmani)

    distmani.load_state_dict(checkpoint['model_state_dict'])
    distmani.eval()
    distmani.to(device)

    imgsz = opt.imgsz

    num_levels = opt.num_levels #5
    if num_levels > 5:
        print("⚠️  max num_levels is 5  ⚠️")
        num_levels = min(5, num_levels)

    if num_levels<1:
        print("⚠️  min num_levels is 1  ⚠️")
        num_levels = max(1, num_levels)


    image_features_collect = []
    labels_lst = []
    mylabels = []
    samples_subfolder = opt.samples_subfolder #100
    labels_flat = []

    all_distortion_keys = []

    fixed_img_files = get_img_files(image_lst=opt.iod_data, samples_subfolder=samples_subfolder)

    print(F"number of images  {len(fixed_img_files)}")
    #input()

    dist_counter = 0
    group_counter = 0
    with torch.no_grad():

        for index, row in enumerate(types_arn):
            print("")
            print(index + 1, row["term"], "->", row["group_id"], row["group_name"])
            img_files = fixed_img_files
            print(F"number of images {len(img_files)}")
            mylabels.append(row["term"])
            all_distortion_keys.extend([row["term"]] * len(img_files))
            for index, img_path in enumerate(tqdm(img_files, total=len(img_files))):
                img_cv = file_to_img(img_path)  # BGR

                if img_cv is None:
                    break

                height_org, width_org = img_cv.shape[:2]
                visu_img = cv2.cvtColor(img_cv.copy(), cv2.COLOR_BGR2RGB)

                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # Padded resize make this after distortion
                # img_padrsize = letterbox_ultra(img_cv, (imgsz, imgsz))

                # img_pil = Image.fromarray(img_padrsize)
                img_torch = transforms.ToTensor()(img_cv)

                img_dist = apply_degradation(
                    image=img_torch,
                    paper=paper_name,
                    term=row["term"],
                    severity=num_levels,
                    mode="original",  # IMPORTANT: native scale of the backend
                )

                img_dist, *_ = letterbox_tensor(img_dist)

                img_dist = img_dist.to(device,
                                       non_blocking=True).float() #/ 255  # r # uint8 to float32, 0-255 to 0.0-1.0

                if img_dist.ndimension() == 3:
                    img_dist = img_dist.unsqueeze(0)

                _, proj_img = distmani.forward(img_dist)

                embeddings = proj_img
                # print(embeddings.shape)
                # input()
                image_features_collect.append(embeddings.squeeze().detach().cpu().numpy())
        # end for all distortions



        # pristine images

        mylabels.append('Pristine')
        #
        dist = "Pristine"
        #img_files = get_img_files(image_lst=opt.iod_data, samples_subfolder=samples_subfolder)
        img_files = fixed_img_files
        if opt.single_distortion:
            labels_lst.append(np.repeat(dist_counter, len(img_files)))
            # Record the distortion key once per sample
            all_distortion_keys.extend([dist] * len(img_files))

        else:
            labels_lst.append(np.repeat(group_counter, len(img_files)))

        dist_counter += 1
        group_counter += 1

        pristine_image_features_collect = []
        for index, img_path in enumerate(tqdm(img_files, total=len(img_files))):
            img_cv = file_to_img(img_path)  # BGR

            if img_cv is None:
                break

            height_org, width_org = img_cv.shape[:2]
            visu_img = cv2.cvtColor(img_cv.copy(), cv2.COLOR_BGR2RGB)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            # Padded resize , but make this after distortion or rather similar to applying distortion
            #img_padrsize = letterbox_ultra(img_cv, (640, 640))

            # print(img.shape)
            # uint8 to float32, 0-255 to 0.0-1.0
            img_torch = transforms.ToTensor()(img_cv)# [0, 1], CxHxW

            img_torch, *_ = letterbox_tensor(img_torch)
            #
            img_torch = img_torch.to(device, non_blocking=True).float() #/ 255 # remove 255 for new checkpoints

            if img_torch.ndimension() == 3:
                img_torch = img_torch.unsqueeze(0)

            # print(F" {img_dist.shape}")

            _, proj_img = distmani.forward(img_torch)

            embeddings = proj_img

            image_features_collect.append(embeddings.squeeze().detach().cpu().numpy())
            pristine_image_features_collect.append(embeddings.squeeze().detach().cpu().numpy())
            labels_flat.append(dist_counter)  # Ein Label pro Bild
        #end for pristine images


        if opt.save_pristine_proto:

            pristine_proto = compute_overall_mean(pristine_image_features_collect)

            pristine_proto_tensor = torch.from_numpy(pristine_proto)

            # save pristine protype as tensor
            torch.save(pristine_proto_tensor, f"{checkpt_folder}/{opt.distmani_folder}/prototype.pt")

            #save
            #np.save("prototype.npy", prototype)
            # Later: Load
            #prototype = np.load("prototype.npy")
            """
            # Save multiple
    
            np.savez("prototypes.npz", overall=prototype, classA=proto_a, classB=proto_b)
            
            # Load
            
            data = np.load("prototypes.npz")
            overall = data["overall"]
            proto_a = data["classA"]
            proto_b = data["classB"]
            """

        #mylabels.append('proto')
        #dist = "Proto"
        #all_distortion_keys.extend([dist])
        # img_files = get_img_files(image_lst=opt.iod_data, samples_subfolder=samples_subfolder)
        #image_features_collect.append(pristine_proto)

        #labels_lst.append(np.repeat(dist_counter, 1))

        #dist_counter += 1


    # --- after collecting image_features_collect and all_distortion_keys ---

    # Stack embeddings

    image_embeddings = np.stack(image_features_collect, axis=0)

    # Safe perplexity (must be < n_samples - 1)

    n_samples = image_embeddings.shape[0]
    perplexity = max(5, min(30, n_samples // 3, n_samples - 2))

    # t-SNE

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        verbose=0,
        max_iter=500
    ).fit_transform(image_embeddings)

    # Plot using your helper (keeps group markers, single colorbar per class)

    out_path = os.path.join(opt.out_folder, opt.out_file_name)
    plot_tsne(
        tsne=tsne,
        all_distortion_keys=all_distortion_keys,
        nice_map=nice_map,
        num_levels=opt.num_levels,
        out_path=out_path,
        hendrycks_groups=paper_groups,
        GROUP_MARKERS=group_markers,
        MARKER_UNICODE=MARKER_UNICODE,
        PRISTINE_KEY=PRISTINE_KEY,
        PRISTINE_GROUP=PRISTINE_GROUP
    )

    #print(F" {image_embeddings.shape}")


    """ end of function----------------------------------------------------------------------------------------------"""


def parse_opt(known=False):
    minga = "/mnt/Data-2TB/"
    minga_2 = "/mnt/Data-512GB/"
    kalle = "/media/ste82041/DATA-2TB/"
    gs007 = "/mnt/15TB-NVME/ste82041/"

    minga_new = "/data/SSD/"
    data_drive = minga_new


    print("checkpoint folder")
    print(CHECKPOINT_ROOT)

    checkpt_folder = checkpoint_path("distortion_manifolds","ablation")

    #checkpt_pt_name= "train_distortion_manifold_bb_5data"
    checkpt_pt_name = "train_distortion_manifold_bb_5data_stable"
    #"coco_10000_50ep_yolov10m_mula_test" #"mixdata_50ep_yolo_mula" # coco_10000_50ep_yolov10m_layer10
    #mixdata_50ep_yolo_mula, coco_10000_50ep_yolov10m_mula

    usb_drive_kalle_02 = "/media/ste82041/Datasets/Datasets"
    minga_new = "/data/SSD/datasets"
    #usb_path = minga_new

    num_levels = 5
    samples_subfolder = 100

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpt_folder', type=str,
                        default=checkpt_folder,
                        help='root path to weights/checkpoints')

    parser.add_argument('--distmani_folder', type=str,
                        default=f"{checkpt_pt_name}",
                        help='root path to weights/checkpoints')


    parser.add_argument('--weights_distmani', type=str,
                        #default="/home/ste82041/python_src/opr_benchmark/checkpoints/distmani/train_faster_distmani/last.pt",
                        default=f"{checkpt_folder}/{checkpt_pt_name}/weights/best.pt",
                        help='initial weights path')

    parser.add_argument('--weights_detector', type=str,
                        ## default="/home/ste82041/python_src/opr_benchmark/checkpoints/distmani/train_faster_distmani/last.pt",
                        default=f'{CHECKPOINT_ROOT}/yolo_ultralytics/coco/yolov10/yolov10m.pt',
                        help='detector weights (YOLO, RT-DETR etc.)')


    parser.add_argument('--iod_data', type=list,
                        default=[
                            f"{data_drive}/datasets/COCO/data/val2017/images",  #
                        ],
                        help='parent path with sub-folder images and labels')

    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help=' image size (pixels)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--show_img', type=bool, default=False, help=' ')

    parser.add_argument('--samples_subfolder', type=int, default=samples_subfolder, help=' ')

    parser.add_argument('--single_distortion', type=bool, default=True, help='single distortion or grouped  ')

    parser.add_argument('--num_levels', type=int, default=num_levels, help='1 - 5  ')

    parser.add_argument('--out_folder', type=str,
                        default="/data/SSD/experiments/sad",
                        help='')

    parser.add_argument('--out_file_name', type=str,
                        # default="/media/ste82041/DATA-2TB/Beckario_main/Beckerio_Paper/img/",
                        default=f"agnolucci_tsne_sad_mula_{num_levels}_val_6datasets_train_coco_{samples_subfolder}",
                        help='')

    #
    parser.add_argument('--save_pristine_proto', type=bool, default=False, help=' saving pristine prototype for AUROC')

    parser.add_argument(
        "--paper",
        type=str,
        choices=["hendrycks", "agnolucci"],
        default="hendrycks",
        help="Which paper taxonomy to use.",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    opt = parse_opt()
    run(opt)
    print("💥 end :: main 💥 ")
    """ end of main -------------------------------------------------------------------------------------------------"""
