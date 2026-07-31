# Data Preparation

NeoForce trains on **[OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData)**, the 5,000-hour open-source subset of NeoData.

## 1. Download the dataset

```bash
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download NeoteAIEmbodied/OpenNeoData --repo-type dataset --local-dir /path/to/OpenNeoData
```

The path you pass here becomes `--data-root` for training and evaluation.

## 2. Dataset layout

OpenNeoData is LeRobot-format, one directory per embodiment:

```
OpenNeoData/<robot>/
  meta/info.json                          feature list, fps, episode counts
  meta/episodes/chunk-*/file-*.parquet    episode index
  videos/<stream>/chunk-XXX/file-XXX.mp4  video streams
```

Two properties matter for the loader. **Many episodes share one mp4**: an episode is the `[from_timestamp, to_timestamp]` slice of it, so seeking is by timestamp rather than by frame index. And each wrist carries three streams, which NeoForce consumes as a single sample:

| Stream | Role |
|---|---|
| `observation.images.<side>_wrist_view` | RGB input, 480×640 |
| `observation.images.<side>_wrist_left_tactile` | force channels 0-2 (left sensor fx, fy, fz) |
| `observation.images.<side>_wrist_right_tactile` | force channels 3-5 (right sensor fx, fy, fz) |

A wrist side is used only when all three streams exist. Sides missing any of them are skipped rather than filled in, because a fabricated stream would become fabricated ground truth. Single-arm embodiments therefore contribute one side instead of two.

## 3. Split

The split ships with this repository as three JSONL files:

| Split | File | Episodes | Used by |
|---|---|---:|---|
| train | `splits/train.jsonl` | 20,000 | `neoforce.train --train-split` |
| val | `splits/val.jsonl` | 2,500 | `neoforce.train --eval-split` |
| test | `splits/test.jsonl` | 2,500 | `neoforce.evaluate --split` |

```
splits/
  train.jsonl / train.csv
  val.jsonl   / val.csv
  test.jsonl  / test.csv
  index.json       episode and frame counts per split
```

Validation and test are equal halves of the held-out set, stratified by category so each
category is split near evenly between them and per-category numbers stay comparable. All
three are disjoint at the episode level.

They are passed straight to the entry points, with no preprocessing step and no dataset scan at startup:

```bash
python -m neoforce.train    ... --train-split splits/train.jsonl --eval-split splits/val.jsonl
python -m neoforce.evaluate ... --split splits/test.jsonl
```

They are also the defaults in `configs/neoforce.json` (`data.index` / `eval_data.index`), so the flags are only needed to point somewhere else.

Each line carries everything the loader needs, with paths relative to `--data-root`:

```json
{
  "category": "fold_towels_from_box-aloha",
  "task": "fold towels from box",
  "robot": "aloha",
  "episode_index": 0,
  "length": 1088,
  "videos": {
    "observation.images.left_wrist_view":          {"path": "aloha/videos/.../file-000.mp4", "from_timestamp": 0.0, "to_timestamp": 173.9},
    "observation.images.left_wrist_left_tactile":  {"path": "...", "from_timestamp": 0.0, "to_timestamp": 173.9},
    "observation.images.left_wrist_right_tactile": {"path": "...", "from_timestamp": 0.0, "to_timestamp": 173.9}
  }
}
```

To build a split over a different selection of episodes, `scripts/build_index.py` walks the LeRobot metadata and emits the same structure:

```bash
python scripts/build_index.py --root /path/to/OpenNeoData --output splits/my_split.json
```

## 4. Force-field statistics

Force fields are inferred online during training from the raw tactile videos. The target scale is fixed once before training, and shared by every run both in training and evaluation.s

Force values are small in absolute terms. Fed straight into a Huber loss with the usual `beta=1`, they sit deep in its quadratic region, where the gradient is proportional to the error and the dense reconstruction branch ends up starved next to the O(1) distillation terms. NeoForce therefore rescales each channel by a high quantile of its in-contact magnitude, putting the targets at O(1).

```bash
python scripts/compute_force_stats.py \
  --config configs/neoforce.json \
  --data-root /path/to/OpenNeoData \
  --split splits/train.jsonl \
  --samples 800 --device cuda \
  --output splits/force_stats_train.json
```

This samples clips through the same conversion model training uses and records, per channel and for both the contact and the all-pixel population, the quantiles q50 … q100 plus mean, std and contact rate. It prints a summary table and writes the JSON that `data.force_stats` points at. The scale actually applied is `force_norm_quantile` (default q99); the rest are kept in the file so the choice can be revisited without rescanning.
