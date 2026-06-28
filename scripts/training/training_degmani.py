import argparse
import gc
import glob
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from degmani.config import DATA_ROOT
from degmani.config.checkpoints_folder import CHECKPOINT_ROOT, checkpoint_path
from degmani.data.data_loader.arniqua_data_loader import ARNIQUA_Base_Dataset
from degmani.losses.losses import nt_xent_loss
from degmani.models.degmani import Degradation_Manifold

TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'
DEFAULT_IMG_PATHS = [f"{DATA_ROOT}/COCO/data/train2017/images"]
DEFAULT_WEIGHTS = f'{CHECKPOINT_ROOT}/yolo_ultralytics/yolov10/coco/yolov10m.pt'


def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)

    dirs = glob.glob(f"{path}{sep}*")
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    indices = [int(match.groups()[0]) for match in matches if match]
    next_index = max(indices) + 1 if indices else 2
    return f"{path}{sep}{next_index}"


def list_yolo_core_modules(
    weights='yolov8n.pt',
    imgsz=640,
    device=None,
    style_layers=None,
    use_train_head=True,
):
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(weights)
    net = yolo.model.to(dev)
    net.eval()
    if use_train_head:
        net.train()

    x = torch.zeros(1, 3, imgsz, imgsz, device=dev)
    hooks = []
    rows = []

    def want_module(index, module):
        if style_layers is None:
            return True
        if not isinstance(style_layers, (list, tuple)):
            return False

        index_filters = [item for item in style_layers if isinstance(item, int)]
        type_filters = [item for item in style_layers if isinstance(item, str)]
        return index in index_filters or module.__class__.__name__ in type_filters

    def shape_str(tensor):
        return 'x'.join(map(str, tensor.shape))

    def hook_fn(index):
        def _hook(module, inputs, output):
            if isinstance(output, (list, tuple)):
                shapes = [shape_str(tensor) for tensor in output if isinstance(tensor, torch.Tensor)]
                out_shape = '[' + ', '.join(shapes) + ']'
            elif isinstance(output, torch.Tensor):
                out_shape = shape_str(output)
            else:
                out_shape = str(type(output))
            rows.append((index, module.__class__.__name__, out_shape))

        return _hook

    for index, module in enumerate(net.model):
        if want_module(index, module):
            hooks.append(module.register_forward_hook(hook_fn(index)))

    with torch.no_grad():
        net(x)

    for hook in hooks:
        hook.remove()

    print('| Index | Type | Output shape(s) |')
    print('|------:|------|------------------|')
    for index, cls_name, shape in rows:
        print(f'| {index} | {cls_name} | {shape} |')


def prep_tensor_visu(x: torch.Tensor):
    img = x[0].detach().cpu().permute(1, 2, 0).contiguous()
    if img.dtype.is_floating_point and img.max() > 1:
        img = (img / 255.0).clamp(0.0, 1.0)
    return img


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train_degmani', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=6,
        help='batch size before contrastive pairing expansion',
    )
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument(
        '--layer_indexes',
        type=int,
        nargs='+',
        default=[0, 1, 3, 5, 7, 10],
        help='Detector layers used by the distortion manifold model',
    )

    parser.add_argument('--hard_neg_examples', type=bool, default=True, help='generate hard negative examples')
    parser.add_argument('--samples_per_dataset', type=int, nargs='+', default=[10000])
    return parser


def print_environment(device: str) -> None:
    python_version = sys.version_info[0]
    print('***********************************************************************************************************')
    print('VCA || Video Content Analysis')
    print(f'torch version {torch.__version__}')
    print(f'python version {python_version}')
    print('***********************************************************************************************************')
    print(f'device: {device}')
    print('checkpoint folder')
    print(CHECKPOINT_ROOT)
    print('data folder')
    print(DATA_ROOT)


def print_options(opt: argparse.Namespace) -> None:
    print('***********************************************************************************************************')
    print('Options:')
    for arg in vars(opt):
        print(f'{arg}: {getattr(opt, arg)}')
    print('***********************************************************************************************************')


def prepare_run_directory(opt: argparse.Namespace) -> Tuple[Path, Path, Path]:
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir = Path(opt.save_dir)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'opt.yaml', 'w') as handle:
        yaml.dump(vars(opt), handle, sort_keys=False)

    return save_dir, weights_dir / 'last.pt', weights_dir / 'best.pt'


def create_dataloader(batch_size: int, samples_per_dataset: Sequence[int], hard_neg_examples: bool) -> DataLoader:
    dataset = ARNIQUA_Base_Dataset(
        DEFAULT_IMG_PATHS,
        samples_per_dataset=samples_per_dataset,
        hard_neg=hard_neg_examples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )


def create_model(device: str, layer_indexes: List[int], embedding_dim: int) -> Degradation_Manifold:
    print(CHECKPOINT_ROOT)
    print(checkpoint_path('yolo_ultralytics', 'yolov10/coco'))
    model = Degradation_Manifold(
        weights=DEFAULT_WEIGHTS,
        layer_idxs=layer_indexes,
        embedding_dim=embedding_dim,
        train_backbone=True,
        device=device,
    )
    model.train()
    return model


def concat_detector_inputs(batch: dict, device: str, hard_neg_examples: bool) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    detector_inputs = [
        batch['imgs_a_dist'].to(device, non_blocking=True).float(),
        batch['imgs_b_dist'].to(device, non_blocking=True).float(),
    ]

    hard_neg_tensors = []
    if hard_neg_examples:
        hard_neg_tensors = [
            batch['imgs_a_dist_ds'].to(device, non_blocking=True).float(),
            batch['imgs_b_dist_ds'].to(device, non_blocking=True).float(),
        ]
        detector_inputs.extend(hard_neg_tensors)

    input_detector = torch.cat(detector_inputs, dim=0)
    if device.startswith('cuda'):
        assert input_detector.is_cuda, 'Input batch is not on CUDA'
    return input_detector, hard_neg_tensors


def forward_embeddings(model, input_detector: torch.Tensor, hard_neg_examples: bool):
    outputs = model.forward_contrastive(input_detector, hard_neg=hard_neg_examples)
    if hard_neg_examples:
        emb_a_dist, emb_b_dist, emb_a_dist_ds, emb_b_dist_ds = outputs
        in_loss_a = torch.cat((emb_a_dist, emb_a_dist_ds), dim=0)
        in_loss_b = torch.cat((emb_b_dist, emb_b_dist_ds), dim=0)
    else:
        emb_a_dist, emb_b_dist = outputs
        in_loss_a = emb_a_dist
        in_loss_b = emb_b_dist
    return in_loss_a, in_loss_b


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    hard_neg_examples: bool,
) -> float:
    total_loss = 0.0
    valid_steps = 0
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=TQDM_BAR_FORMAT)

    for _, batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        input_detector, _ = concat_detector_inputs(batch, device, hard_neg_examples)
        in_loss_a, in_loss_b = forward_embeddings(model, input_detector, hard_neg_examples)
        loss = nt_xent_loss(in_loss_a, in_loss_b)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        valid_steps += 1

    if valid_steps == 0:
        return float('inf')
    return total_loss / valid_steps


def save_checkpoint(last_path: Path, best_path: Path, epoch: int, model, optimizer, layer_indexes: List[int], opt, epoch_loss: float, best_loss: float) -> float:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'layer_indexes': layer_indexes,
        'train_args': vars(opt),
    }
    torch.save(checkpoint, last_path)
    if epoch_loss < best_loss:
        torch.save(checkpoint, best_path)
        return epoch_loss
    return best_loss


def main() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_environment(device)

    parser = build_parser()
    opt = parser.parse_args()
    print_options(opt)

    save_dir, last_path, best_path = prepare_run_directory(opt)
    print(f'save to {save_dir}')

    loader = create_dataloader(
        batch_size=opt.batch_size,
        samples_per_dataset=opt.samples_per_dataset,
        hard_neg_examples=opt.hard_neg_examples,
    )
    model = create_model(device, opt.layer_indexes, opt.embedding_dim)
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)

    total_steps = opt.epochs * len(loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_loss = float('inf')
    for epoch in range(opt.epochs):
        epoch_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            hard_neg_examples=opt.hard_neg_examples,
        )
        print(f'Epoch {epoch + 1}/{opt.epochs}, Average Loss: {epoch_loss:.8f}')
        best_loss = save_checkpoint(
            last_path=last_path,
            best_path=best_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            layer_indexes=opt.layer_indexes,
            opt=opt,
            epoch_loss=epoch_loss,
            best_loss=best_loss,
        )

    print('end :: main')


if __name__ == '__main__':
    main()
