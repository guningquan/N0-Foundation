# Evaluation

NeoForce is evaluated on the held-out `splits/test.jsonl` split (2,500 episodes) with three metrics: **force MAE**, **force RMSE**, and **contact mIoU**.

## Quick start

```bash
bash scripts/evaluate.sh /path/to/OpenNeoData outputs/neoforce/neoforce_last.pt eval_result/neoforce.json
```

Explicit form:

```bash
python -m neoforce.evaluate \
  --checkpoint outputs/neoforce/neoforce_last.pt \
  --config configs/neoforce.json \
  --data-root /path/to/OpenNeoData \
  --split splits/test.jsonl \
  --output eval_result/neoforce.json
```

## Metrics

| Metric | Definition |
|---|---|
| **MAE** | mean absolute error of the predicted force field over ground-truth contact pixels, relative to the per-channel training scale, averaged over the 6 channels |
| **RMSE** | root mean squared error over the same pixels, on the same scale |
| **mIoU** | IoU between the predicted contact mask (`sigmoid(logits) ≥ 0.5`) and the ground-truth mask, averaged over channels that appear in the split |

The 6 channels are `[left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]`. Per-channel values are reported alongside the averages; `fz` (pressure) generally carries larger magnitudes than the shear channels, so the mean across channels is dominated by it.

## Coverage and cost

The test split expands to a large number of clips, and every clip runs the tactile conversion model to produce its ground truth. Two knobs control cost:

| Flag | Default | Effect |
|---|---|---|
| `--sample-stride` | `256` | Clip start spacing in frames. The default gives broad episode coverage at a tractable cost; lower it for a denser pass |
| `--max-batches` | `0` (all) | Hard cap on batches, for a quick check |

Both are recorded in the output JSON. A run with a different stride is not directly comparable to one at the default.

## Output

```
channel               MAE           RMSE      IoU
left_fx      X.XXXXXXe-XX   X.XXXXXXe-XX   0.XXXX
left_fy      ...
...

  MAE  : X.XXXXXXe-XX
  RMSE : X.XXXXXXe-XX
  mIoU : 0.XXXX
```

The JSON report carries the same numbers plus the checkpoint path, split, and mask threshold:

```json
{
  "checkpoint": "outputs/neoforce/neoforce_last.pt",
  "split": "splits/test.jsonl",
  "mask_threshold": 0.5,
  "metrics": {
    "mae": 0.0, "rmse": 0.0, "miou": 0.0,
    "mae_by_channel": {"left_fx": 0.0, "...": 0.0},
    "rmse_by_channel": {"...": 0.0},
    "iou_by_channel": {"...": 0.0},
    "clips": 0
  }
}
```
