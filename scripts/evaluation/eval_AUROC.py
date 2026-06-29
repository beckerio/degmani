import argparse
import gc
import random
import sys
from itertools import cycle

from matplotlib.colors import ListedColormap
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns

from degmani.degmani_utils.io import file_to_img, load_img_from_folder_list, yaml_load
from degmani.degmani_utils.detection_utils import letterbox_ultra

from degmani.config.checkpoints_folder import CHECKPOINT_ROOT, checkpoint_path
from degmani.models.degmani import Degradation_Manifold

from degmani.degmani_utils.eval_utils import embedding_distances
from degmani.degmani_utils.ood_plotting import plot_ood_scores
from degmani.degmani_utils.ood_metrics import area_under_gaussian_at_left, area_under_gaussian_at_right


IMAGE_SIZE = (640, 640)
EPSILON = 1e-12


def as_list(value):
    return value if isinstance(value, list) else [value]


def load_paired_image_paths(iod_data, ood_data, number_samples):
    iod_dirs = as_list(iod_data)
    ood_dirs = as_list(ood_data)
    assert len(iod_dirs) == len(ood_dirs), "iod_data and ood_data must contain the same number of folders."

    img_files = []
    img_files_ood = []
    labels = []
    label_names = []
    dataset_counter = 0

    for iod_dir, ood_dir in zip(iod_dirs, ood_dirs):
        print(f"loading paired images from\n  ID : {iod_dir}\n  OOD: {ood_dir} (num_samples={number_samples})")

        iod_list = load_img_from_folder_list(iod_dir)
        ood_list = load_img_from_folder_list(ood_dir)
        assert len(iod_list) == len(ood_list), (
            f"folders have different lengths: {iod_dir} ({len(iod_list)}), "
            f"{ood_dir} ({len(ood_list)})"
        )

        n_available = len(iod_list)
        if n_available == 0:
            raise ValueError(f"No paired images found in {iod_dir} and {ood_dir}.")

        k = min(n_available, number_samples)
        if k < number_samples:
            print(f"Warning: only {n_available} paired images available; using {k} samples.")

        idxs = random.sample(range(n_available), k=k)
        img_files.extend(iod_list[i] for i in idxs)
        img_files_ood.extend(ood_list[i] for i in idxs)

        labels.append(np.repeat(dataset_counter, k))
        label_names.append("ID")
        dataset_counter += 1

        labels.append(np.repeat(dataset_counter, k))
        label_names.append("OOD")
        dataset_counter += 1

    print(f"number of images {len(img_files)} (ID), {len(img_files_ood)} (OOD)")
    return img_files, img_files_ood, labels, label_names, dataset_counter


def load_image_tensor(img_path, device):
    img_cv = file_to_img(img_path)
    if img_cv is None:
        raise ValueError(f"Could not load image: {img_path}")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_rgb = letterbox_ultra(img_rgb, IMAGE_SIZE)
    img_pil = Image.fromarray(img_rgb)
    img = transforms.ToTensor()(img_pil).to(device, non_blocking=True).float()

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def score_images(img_files, distmani, proto_np, device):
    scores = []
    embeddings = []

    for img_path in tqdm(img_files, total=len(img_files)):
        img = load_image_tensor(img_path, device)
        _, proj_img = distmani.forward(img)
        embedding = proj_img.squeeze().detach().cpu().numpy()
        embeddings.append(embedding)

        dist = embedding_distances(proto_np, embedding)
        scores.append(dist["cosine_distance"])

    return scores, embeddings


def plot_embeddings(image_features, label_arrays, label_names, prototype_label):
    labels = np.concatenate(label_arrays)
    image_embeddings = np.stack(image_features, axis=0)

    perplexity = min(30, max(1, len(image_embeddings) - 1))
    tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=perplexity, max_iter=300).fit_transform(image_embeddings)
    colors = (["cornflowerblue", "violet"] * ((len(label_names) // 2) + 1))[:len(label_names)]
    colors[-1] = "cyan"
    cmap = ListedColormap(colors)
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())

    plt.figure(figsize=(10, 4))
    marker_cycler = cycle(["o", "s", "*", "^", "v", "D"])
    for label in np.unique(labels):
        mask = labels == label
        marker = next(marker_cycler)
        marker_size = 200 if label == prototype_label else 40
        plt.scatter(
            tsne[mask, 0],
            tsne[mask, 1],
            c=labels[mask],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=marker_size,
            edgecolors="black",
            label=str(label),
        )

    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(len(label_names) + 1) - 0.5).set_ticks(
        np.arange(len(label_names)),
        labels=label_names,
    )
    plt.xlabel("t-SNE $x$ ")
    plt.ylabel("t-SNE $y$ ")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.close("all")


def analyse(opt, device="cuda"):
    number_samples = opt.num_samples_dataset
    img_files, img_files_ood, labels_lst, mylabels, dataset_counter = load_paired_image_paths(
        opt.iod_data,
        opt.ood_data,
        number_samples,
    )

    checkpt_folder = opt.checkpt_folder

    training_settings = yaml_load(f"{checkpt_folder}/{opt.distmani_folder}/opt.yaml")

    print("Training settings")
    for key, value in training_settings.items():
        print(f"{key}: {value}")

    weights = opt.weights_detector
    layer_indexes = training_settings['layer_indexes']

    distmani = Degradation_Manifold(weights=weights,
                                      layer_idxs=layer_indexes,
                                      embedding_dim=training_settings['embedding_dim'],
                                      train_backbone=True,
                                      device=device)

    checkpoint = torch.load(opt.weights_distmani, map_location=device)

    distmani.load_state_dict(checkpoint['model_state_dict'])
    distmani.eval()
    distmani.to(device)

    print("load protoype ")
    prototype = torch.load(f"{checkpt_folder}/{opt.distmani_folder}/prototype.pt", map_location="cpu")
    proto_np = prototype.detach().cpu().numpy()

    image_features_collect = []

    with torch.no_grad():
        print("\n id samples ")
        iod_scores, id_features = score_images(img_files, distmani, proto_np, device)
        image_features_collect.extend(id_features)

        print("\n ood samples ")
        ood_scores, ood_features = score_images(img_files_ood, distmani, proto_np, device)
        image_features_collect.extend(ood_features)

    if opt.visu_embeddings:
        mylabels.append('Pristine \n Prototype')
        proto_label = dataset_counter
        labels_lst.append(np.array([proto_label]))
        image_features_collect.append(proto_np)
        plot_embeddings(image_features_collect, labels_lst, mylabels, proto_label)

    print("Compare statistics")
    id_scores = np.array(iod_scores)
    ood_scores = np.array(ood_scores)

    id_mean = id_scores.mean()
    id_std = id_scores.std() + EPSILON

    id_score_z = np.abs((id_scores - id_mean) / id_std)
    ood_score_z = np.abs((ood_scores - id_mean) / id_std)

    labels = np.array([0] * len(id_score_z) + [1] * len(ood_score_z))
    scores = np.concatenate([id_score_z, ood_score_z])

    auroc = roc_auc_score(labels, scores)
    print(f"AUROC (z-distance): {auroc:.4f}")

    iod_scores_np = id_scores
    ood_scores_np = ood_scores

    mu_iod_score = np.mean(iod_scores_np)
    sigma_iod_score = np.std(iod_scores_np)

    mu_ood_score = np.mean(ood_scores_np)
    sigma_ood_score = np.std(ood_scores_np)

    min_score = min(min(iod_scores_np), min(ood_scores_np))
    max_score = max(max(iod_scores_np), max(ood_scores_np))

    if opt.z_score:
        iod_scores_normed = id_score_z
        ood_scores_normed = ood_score_z
    else:
        iod_scores_normed = id_scores
        ood_scores_normed = ood_scores

    print(iod_scores_normed.shape, ood_scores_normed.shape)
    plot_ood_scores(id_data=iod_scores_normed, ood_data=ood_scores_normed, save=False)

    bins = np.linspace(min_score, max_score, 50)
    t = np.linspace(min_score, max_score, 1000)
    false_positive = area_under_gaussian_at_right(t, mu_iod_score, sigma_iod_score + EPSILON)
    false_negative = area_under_gaussian_at_left(t, mu_ood_score, sigma_ood_score + EPSILON)

    threshold = t[np.argmin(false_negative + false_positive)]
    print(F"mean  {mu_iod_score} +-std{sigma_iod_score}, {mu_ood_score} +- {sigma_ood_score}")

    print(F"threshold  {threshold}")

    plt.figure()
    plt.hist(iod_scores_np, bins=bins, alpha=0.5, label='ID Scores', color="cornflowerblue")
    plt.hist(ood_scores_np, bins=bins, alpha=0.5, label='OOD Scores', color="violet")
    plt.legend()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(np.array(id_score_z), label='ID scores', color="cornflowerblue", fill=True, alpha=0.5)
    sns.kdeplot(np.array(ood_score_z), label='OoD scores', color="violet", fill=True, alpha=0.5)

    plt.legend()
    plt.title("Score distributions")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.grid(True)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8, linestyle=':')
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    plt.close('all')


if __name__ == '__main__':
    PYTHON_VERSION = sys.version_info[0]
    print("***********************************************************************************************************")
    print(f"torch version {torch.__version__}")
    print(f"python version {PYTHON_VERSION}")
    print("***********************************************************************************************************")
    gc.collect()
    torch.cuda.empty_cache()  # clear GPU memory, may help reduce CUDA out of memory errors

    print("checkpoint folder")
    print(CHECKPOINT_ROOT)

    checkpt_folder = checkpoint_path("distortion_manifolds", "ablation")

    checkpt_pt_name = "train_distortion_manifold_bb_5data"
    data_drive = "/data/SSD/"

    level = 5  # [1, ..., 5]

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpt_folder', type=str,
                        default=checkpt_folder,
                        help='root path to weights/checkpoints')

    parser.add_argument('--distmani_folder', type=str,
                        default=f"{checkpt_pt_name}",
                        help='root path to weights/checkpoints')

    parser.add_argument('--weights_distmani', type=str,
                        default=f"{checkpt_folder}/{checkpt_pt_name}/weights/best.pt",
                        help='initial weights path')

    parser.add_argument('--weights_detector', type=str,
                        default=f'{CHECKPOINT_ROOT}/yolo_ultralytics/coco/yolov10/yolov10m.pt',
                        help='detector weights (YOLO, RT-DETR etc.)')

    parser.add_argument('--iod_data', nargs='+',
                        default=[
                            f"{data_drive}datasets/COCO/data/val2017/images",
                        ],
                        help='parent path with sub-folder images and labels')

    parser.add_argument('--ood_data', nargs='+',
                        default=[
                            f"{data_drive}datasets/COCO/data/val2017/images_c{level}",
                        ],
                        help='parent path with sub-folder images and labels')

    parser.add_argument('--level', type=int, default=level, help=' ')

    parser.add_argument('--num_samples_dataset', type=int,
                        default=5000,  # COCO
                        help=' ')

    parser.add_argument('--augment', type=bool, default=False, help=' ')

    parser.add_argument('--corruption_name', type=str, default="gaussian_noise", help=" ")
    parser.add_argument('--corruption_level', type=int, default=5, help=" ")

    parser.add_argument('--visu_embeddings', type=bool, default=True)

    parser.add_argument('--ood_reco_type', type=str, default="ce", help="bce, ce, mse ")

    parser.add_argument('--save_dir', type=str, default="/data/SSD/experiments/test", help="")

    parser.add_argument('--z_score', type=bool, default=False, help=' ')

    opt = parser.parse_args()
    print("***********************************************************************************************************")
    print("Options:")
    for arg in vars(opt):
        print(F"{arg}: {getattr(opt, arg)}")
    print("***********************************************************************************************************")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyse(opt, device)
    print("💥 That's all folks 💥 ")
