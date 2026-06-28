"""Save a post-training pristine prototype for DEGMANI.

This is a simplified standalone script for computing the pristine prototype
after training. It can also save a prototype on a selected degradation subset.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional

import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from degmani.config.checkpoints_folder import CHECKPOINT_ROOT, checkpoint_path
from degmani.degmani_utils.eval_utils import compute_overall_mean
from degmani.degmani_utils.img_utils import letterbox_tensor
from degmani.degmani_utils.io import file_to_img, load_img_from_folder_list, yaml_load
from degmani.models.degmani import Degradation_Manifold
"""
https://github.com/beckerio/imdeg
"""
from imdeg import apply_degradation, list_paper_types
from imdeg.config.papers import PAPER_NAME_MAP, PaperSelection


def sample_image_files(image_sources: List[str], samples_per_source: int) -> List[str]:
    image_files: List[str] = []
    for image_source in image_sources:
        print(f'loading images from {image_source}')
        available_files = load_img_from_folder_list(image_source)
        if not available_files:
            raise FileNotFoundError(f'No images found in {image_source}')
        image_files.extend(random.choices(available_files, k=samples_per_source))
    return image_files


def load_model(opt, device: str):
    training_settings = yaml_load(f'{opt.checkpt_folder}/{opt.distmani_folder}/opt.yaml')
    print('Training settings')
    for key, value in training_settings.items():
        print(f'{key}: {value}')

    layer_indexes = training_settings.get('layer_indexes', [0, 1, 3, 5, 7, 10])
    model = Degradation_Manifold(
        weights=opt.weights_detector,
        layer_idxs=layer_indexes,
        embedding_dim=training_settings['embedding_dim'],
        train_backbone=True,
        device=device,
    )

    checkpoint = torch.load(opt.weights_distmani, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model


def clamp_severity(severity: int) -> int:
    if severity < 1:
        return 1
    if severity > 5:
        return 5
    return severity


def validate_term(paper_name: str, term: str) -> None:
    available_terms = {row['term'] for row in list_paper_types(paper_name)}
    if term not in available_terms:
        available_sorted = ', '.join(sorted(available_terms))
        raise ValueError(f"Unknown term '{term}' for paper '{paper_name}'. Available terms: {available_sorted}")


def image_to_embedding(model, img_path: str, device: str, degradation: Optional[dict]) -> Optional[torch.Tensor]:
    img_cv = file_to_img(img_path)
    if img_cv is None:
        return None

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb)

    if degradation is not None:
        img_tensor = apply_degradation(
            image=img_tensor,
            paper=degradation['paper_name'],
            term=degradation['term'],
            severity=degradation['severity'],
            mode='original',
        )

    img_tensor, *_ = letterbox_tensor(img_tensor)
    img_tensor = img_tensor.to(device, non_blocking=True).float()
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    _, embedding = model.forward(img_tensor)
    return embedding.squeeze().detach().cpu()


def compute_prototype(model, image_files: List[str], device: str, degradation: Optional[dict]) -> torch.Tensor:
    # Unlike the paper-specific prototypes, this script saves a post-training DEGMANI
    # prototype computed from the model's embeddings on either pristine images or a
    # selected degradation subset.
    embeddings = []
    for img_path in tqdm(image_files, total=len(image_files)):
        embedding = image_to_embedding(model, img_path, device, degradation)
        if embedding is None:
            continue
        embeddings.append(embedding.numpy())

    if not embeddings:
        raise RuntimeError('No embeddings could be computed from the selected images.')

    prototype = compute_overall_mean(embeddings)
    return torch.from_numpy(prototype)


def build_output_path(opt, degradation: Optional[dict]) -> Path:
    if opt.output is not None:
        return Path(opt.output)

    if degradation is None:
        file_name = 'prototype.pt'
    else:
        safe_term = degradation['term'].replace('/', '_').replace(' ', '_')
        file_name = f"prototype_{opt.paper}_{safe_term}_severity{degradation['severity']}.pt"
    return Path(opt.checkpt_folder) / opt.distmani_folder / file_name


def parse_opt() -> argparse.Namespace:
    default_checkpt_folder = checkpoint_path('distortion_manifolds', 'ablation')
    default_distmani_folder = 'train_distortion_manifold_bb_5data_stable'
    default_data_root = '/data/SSD/datasets'

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt_folder', type=str, default=default_checkpt_folder, help='root path to checkpoints')
    parser.add_argument('--distmani_folder', type=str, default=default_distmani_folder, help='checkpoint run folder')
    parser.add_argument(
        '--weights_distmani',
        type=str,
        default=f'{default_checkpt_folder}/{default_distmani_folder}/weights/best.pt',
        help='trained manifold checkpoint',
    )
    parser.add_argument(
        '--weights_detector',
        type=str,
        default=f'{CHECKPOINT_ROOT}/yolo_ultralytics/coco/yolov10/yolov10m.pt',
        help='detector backbone weights',
    )
    parser.add_argument(
        '--iod_data',
        nargs='+',
        default=[f'{default_data_root}/COCO/data/val2017/images'],
        help='one or more image folders to sample from',
    )
    parser.add_argument('--samples_subfolder', type=int, default=100, help='number of sampled images per folder')
    parser.add_argument('--paper', choices=['hendrycks', 'agnolucci'], default='agnolucci', help='paper taxonomy')
    parser.add_argument('--term', type=str, default=None, help='degradation term for degraded prototype; omit for pristine')
    parser.add_argument('--severity', type=int, default=5, help='degradation severity in [1, 5]')
    parser.add_argument('--output', type=str, default=None, help='output .pt path; defaults into checkpoint run folder')
    parser.add_argument('--seed', type=int, default=0, help='random seed for image sampling')
    return parser.parse_args()


def main() -> None:
    opt = parse_opt()
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    degradation = None
    if opt.term:
        paper_enum = PaperSelection[opt.paper.upper()]
        paper_name = PAPER_NAME_MAP[paper_enum]
        validate_term(paper_name, opt.term)
        degradation = {
            'paper_name': paper_name,
            'term': opt.term,
            'severity': clamp_severity(opt.severity),
        }
        print(f"computing degraded prototype for paper='{opt.paper}', term='{opt.term}', severity={degradation['severity']}")
    else:
        print('computing pristine prototype')

    image_files = sample_image_files(opt.iod_data, opt.samples_subfolder)
    print(f'number of sampled images: {len(image_files)}')

    model = load_model(opt, device)
    prototype = compute_prototype(model, image_files, device, degradation)

    output_path = build_output_path(opt, degradation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototype, output_path)
    print(f'saved prototype to {output_path}')
    print(f'prototype shape: {tuple(prototype.shape)}')


if __name__ == '__main__':
    main()
