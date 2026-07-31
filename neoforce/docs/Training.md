# Training

## Quick start

```bash
bash scripts/train.sh /path/to/OpenNeoData 8 outputs/neoforce
```

That wraps `torchrun` over `neoforce.train`. The explicit form:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  -m neoforce.train \
  --config configs/neoforce.json \
  --data-root /path/to/OpenNeoData \
  --train-split splits/train.jsonl \
  --eval-split  splits/val.jsonl \
  --force-stats splits/force_stats_train.json \
  --output-dir outputs/neoforce \
  --dinov2-checkpoint /path/to/dinov2_vitb14_pretrain.pth   # optional warm start
```

**Prerequisite:** `splits/force_stats_train.json` must exist; see [Data Preparation](./DataPreparation.md#4-force-field-statistics). Training refuses to start without it when `force_norm_mode` is `quantile`, because the alternative is silently training on unnormalized targets.

## Training Pipeline

Each sample is a **4-frame clip at 30 fps** (0.13 s) from one wrist of one episode:

1. Decode 4 RGB frames and 4 frames from each of the two tactile videos.
2. Run the tactile frames through the frozen tactile conversion model: bird-view correction, optical flow against the episode's first frame, the conversion network, then masking and scaling. This produces `force (4, 6, 360, 640)` and the model's own contact mask, used directly as ground truth.
3. Normalize the force by the per-channel scale from the statistics file.
4. Forward through the encoder; reconstruct force and contact mask at full resolution; regress the per-frame global force; predict masked patch latents against an EMA teacher.

## Objective

| Term | Weight | Description |
|---|---:|---|
| `aux` | 1.0 | the same force, mask and direction terms at the decoder's 1/8, 1/4 and 1/2 stages, averaged |
| `force` | 10.0 | Huber on the force field, inside ground-truth contact, normalized per channel by contact count |
| `force_global` | 2.0 | per-frame channel mean, regressed from the temporal context |
| `force_delta` | 2.0 | frame-to-frame force difference, inside contact |
| `force_no_contact` | 0.05 | force outside contact (target is exactly zero) |
| `mask` | 0.5 | contact mask: BCE + Dice |
| `direction` | 1.0 | `1 − cos` between predicted and true force vectors, inside contact |
| `jepa_t` | 1.0 | masked tactile patch latents regressed onto the EMA teacher (Smooth-L1 on LayerNorm'd targets) |
| `contrast` | 0.05 | image-touch InfoNCE between the two CLS projections |
| `dino_v`, `dino_t`, `dino_cross` | 0.05 | CLS-level and cross-modal distillation onto the EMA teacher |
| `jepa_v` | 0.05 | the vision-side counterpart of `jepa_t`, on the visual patch latents |

## Configuration

Everything lives in `configs/neoforce.json`.

| Key | Default | Notes |
|---|---|---|
| `model.backbone` | `ViT-B-img360x640-patch20` | One string sets width/depth/heads, resolution and patch size |
| `model.decoder_start_stride` | `8` | Decoding starts at out/8 and doubles to full resolution |
| `model.decoder_stage_channels` | `[128, 64, 32, 16]` | Width shrinks as resolution grows |
| `model.decoder_deep_supervision` | `true` | Prediction head and loss at every stage |
| `data.seq_len` | `4` | Frames per clip |
| `data.sample_stride` | `8` | Clip start spacing, in frames |
| `data.sensor_device` | `cpu` | Device for the tactile conversion model **inside the dataloader workers** |
| `train.batch_size` | `1` | Per rank; effective batch = `batch_size × num_gpus` |
| `train.num_workers` | `12` | Per rank |
| `train.amp_dtype` | `bf16` | fp16 additionally enables the grad scaler |
| `train.touch_dropout` | `0.0` | See the note below |
| `train.max_steps` | `100000` | |
| `train.eval_every` | `2000` | Validation cadence; reports MAE, RMSE and mIoU on `splits/val.jsonl` |
| `train.save_every` | `5000` | Each checkpoint holds model + EMA teacher + optimizer state |

