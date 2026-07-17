# degmani

Code for the paper **"Self-Aware Object Detection via Degradation Manifolds"**.

This is research code for paper reproducibility and selected experiments, not a polished Python package.

`degmani` contains training, prototype computation, and analysis code for learning degradation-aware detector embeddings. The method augments an object detector backbone with a lightweight embedding head and uses a pristine prototype as a nominal reference for degradation-aware monitoring.

## Core idea:
- learn a degradation manifold in detector feature space rather than relying only on detector confidence
- train with degraded image views to structure embeddings by degradation type and severity
- estimate a pristine prototype from clean embeddings as a nominal operating point
- use geometric deviation from that prototype as a self-awareness signal

## Repository Layout

- `src/degmani`: package source code
- `src/degmani/models`: manifold model definitions
- `src/degmani/data`: dataset and loader code
- `src/degmani/degmani_utils`: distortions, IO, feature extraction, and helpers
- `src/degmani/losses`: contrastive loss functions
- `scripts/training/training_degmani.py`: main training script
- `scripts/training/save_prototype_degmani.py`: post-training prototype export
- `scripts/analyse/`: analysis and visualization scripts
- `paper/`: paper material

## Installation

Install from the repository root:

```bash
pip install -e .
```

Notes:
- Dependencies are declared in `pyproject.toml`.
- Install the PyTorch build that matches your CUDA setup.
- Some script defaults assume local checkpoint and dataset layouts and may need to be adapted.

## Training

Run training from the repository root:

```bash
python scripts/training/training_degmani.py
```

## Post-Training Prototype Saving

Compute the pristine prototype:

```bash
python scripts/training/save_prototype_degmani.py
```

Compute a prototype on a selected degradation subset:

```bash
python scripts/training/save_prototype_degmani.py \
  --paper agnolucci \
  --term gaublur \
  --severity 5
```

## Analysis

The scripts under `scripts/analyse/` support experiment workflows such as:
- t-SNE visualization of embedding spaces
- paper-specific degradation selection
- prototype estimation from selected image sets
- embedding inspection under pristine and degraded inputs
- `imdeg` is an optional dependency used only for selected visualization and prototype-analysis workflows, not for the core training pipeline.
- To enable those workflows, install the optional visualization dependency separately `https://github.com/beckerio/imdeg`.

## Notes
- Parts of the code still reflect active paper experimentation and checkpoint-specific assumptions.
- The project uses a `src` layout.
- Imports should use `degmani...` package paths.


## Citation
If you use this code in academic work, please consider to cite the following paper:
```bibtex
@misc{becker2026selfawareobjectdetectiondegradation,
      title={Self-Aware Object Detection via Degradation Manifolds}, 
      author={Stefan Becker and Simon Weiss and Wolfgang Hübner and Michael Arens},
      year={2026},
      eprint={2602.18394},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.18394}, 
}
```

## Acknowledgement:
- Agnolucci et al.: [GitHub](https://github.com/miccunifi/ARNIQA) - [Paper](https://ieeexplore.ieee.org/document/10483567)  
- Hendrycks & Dietterich: [GitHub](https://github.com/hendrycks/robustness) - [Paper](https://openreview.net/forum?id=HJz6tiCqYm)
- Harakeh et al.: [GitHub](https://github.com/asharakeh/probdet) - [Paper](https://openreview.net/forum?id=YLewtnvKgR7)
- Oksuz et al.: [GitHub](https://github.com/fiveai/saod) - [Paper](https://ieeexplore.ieee.org/document/10203797)
- Michaelis et al.: [GitHub](https://github.com/bethgelab/robust-detection-benchmark) - [Paper](https://openreview.net/forum?id=ryljMpNtwr)






  
