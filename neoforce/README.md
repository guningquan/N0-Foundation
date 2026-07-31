<h1 align="center">NeoForce: A Unified Force Tactile Representation Model</h1>

> **NeoForce**: the unified force-based tactile representation of **𝒩<sub>0</sub>-Foundation**.<br>
> [Technical Report](https://research.neoteai.com/assets/n0-foundation-paper.pdf) · [Project Website](http://research.neoteai.com/n0-foundation) · [Weights](https://huggingface.co/NeoteAI/NeoForce) · [OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData)

Tactile hardware is fragmented, so a model built on one signal format is bound to the device that produced it. NeoForce instead describes every tactile observation as a **dense three-axis force field** over the sensing surface and learns a temporally structured representation on top of it.

The model is a **joint visual-tactile ViT**: RGB frames and dual-sensor force fields are patched onto one shared token grid and run through a single trunk, so the modalities attend to each other directly instead of being fused after the fact. A temporal transformer over per-frame CLS pairs supplies context, and a progressive decoder reconstructs the force field and contact mask. Training is supervised jointly by force-field reconstruction and masked latent prediction against an EMA teacher.

## Highlights

- **Hardware-agnostic target.** Every observation is a dense `(6, H, W)` force field, `[left fx, fy, fz, right fx, fy, fz]`, rather than a device-specific image.
- **Full-resolution output.** Decoding starts at 1/8 resolution and doubles to 360×640, with a prediction head and a loss at every stage. One bilinear jump from the token grid cannot represent a contact boundary narrower than a patch; a supervised ladder can.

## Installation

See the [Installation Guide](./docs/Installation.md) for the environment, CUDA-matched PyTorch, and the tactile conversion model.

## Data Preparation

See the [Data Preparation Guide](./docs/DataPreparation.md) for downloading OpenNeoData, the train/test split shipped in `splits/`, and the one-off force-field statistics pass.

## Training

See the [Training Guide](./docs/Training.md).

```bash
bash scripts/train.sh /path/to/OpenNeoData 8 outputs/neoforce
```

## Evaluation

See the [Evaluation Guide](./docs/Evaluation.md). Reports force **MAE** and **RMSE** over contact pixels, and contact-mask **mIoU**.

```bash
bash scripts/evaluate.sh /path/to/OpenNeoData outputs/neoforce/neoforce_last.pt
```

## Repository Layout

```text
neoforce/
  models/encoder.py          joint visual-tactile ViT + progressive decoder
  models/jepa.py             JEPA predictor and I-JEPA multi-block masking
  data/openneo.py            OpenNeoData reader with online force-field GT
  data/force_norm.py         per-channel force normalization and its statistics
  visuo_tactile_conversion/  frozen tactile conversion model
  losses.py                  reconstruction + distillation objective
  train.py / evaluate.py     entry points
weights/neoforce.pt          trained weights, student + EMA teacher (see docs/Installation.md)
configs/neoforce.json        the released configuration
scripts/                     index building, force statistics, train.sh, evaluate.sh
splits/            train / test split (JSONL)
docs/                        Installation, Data Preparation, Training, Evaluation
```

## Model

| | |
|---|---|
| Backbone | ViT-B, 12 layers, 768-dim, patch 20 |
| Input | 4 frames @ 30 fps, RGB 360×640 + tactile representation `(6, 360, 640)` |
| Output | tactile representation `(6, 360, 640)`, contact mask `(6, 360, 640)`, global force `(6,)` |

Weights: [`NeoteAI/NeoForce`](https://huggingface.co/NeoteAI/NeoForce) on Hugging Face. See the [Installation Guide](./docs/Installation.md#4-place-the-weights) for where each file goes.

## 👍 Citation

```bibtex
@misc{n0foundation,
      title={N0-Foundation: Towards the Age of Tactile Intelligence},
      author={NeoteAI Team and TEAI Team},
      year={2026},
      url={https://research.neoteai.com/assets/n0-foundation-paper.pdf},
      note={Technical Report}
}
```

## 🏷️ License

Distributed under the terms of the [MIT License](./LICENSE).
