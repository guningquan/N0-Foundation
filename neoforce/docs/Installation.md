# Installation

## Requirements

- Linux with an NVIDIA GPU
- Python 3.10
- CUDA-enabled PyTorch ≥ 2.3

## Setup

### 1. Create an environment

```bash
conda create -n neoforce python=3.10 -y
conda activate neoforce
```

### 2. Install PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the remaining dependencies

```bash
cd neoforce
pip install -r requirements.txt
```

### 4. Place the weights

Two weight files are needed. They are distributed separately from the source, and each has
a fixed path the code looks in:

Both live in the [`NeoteAI/NeoForce`](https://huggingface.co/NeoteAI/NeoForce) model repository:

| File | In the model repo | Place at | Size |
|---|---|---|---|
| NeoForce weights | `neoforce/neoforce.pt` | `weights/neoforce.pt` | 1.25 GB |
| Tactile conversion model | `visuo_tactile_conversion/visuo_tactile_conversion.pt` | `neoforce/visuo_tactile_conversion/model/visuo_tactile_conversion.pt` | 49 MB |

```bash
mkdir -p weights neoforce/visuo_tactile_conversion/model

curl -L -o weights/neoforce.pt \
  https://huggingface.co/NeoteAI/NeoForce/resolve/main/neoforce/neoforce.pt
curl -L -o neoforce/visuo_tactile_conversion/model/visuo_tactile_conversion.pt \
  https://huggingface.co/NeoteAI/NeoForce/resolve/main/visuo_tactile_conversion/visuo_tactile_conversion.pt
```

Or clone the whole model repository and copy the two files into place:

```bash
git clone https://huggingface.co/NeoteAI/NeoForce
cp NeoForce/neoforce/neoforce.pt weights/
cp NeoForce/visuo_tactile_conversion/visuo_tactile_conversion.pt \
   neoforce/visuo_tactile_conversion/model/
```

```
neoforce/
  weights/
    neoforce.pt                       <-- NeoForce weights go here
  neoforce/visuo_tactile_conversion/
    conversion.py                     force-field conversion
    batch_flow/optical_flow_dis.py    optical flow
    model/
      network.py                      network definition (ships with the source)
      visuo_tactile_conversion.pt     <-- conversion weights go here
```

The conversion model is resolved relative to its own package, so that path is fixed.
`weights/neoforce.pt` is only the default the examples use; both entry points accept an
explicit `--checkpoint`.

### 5. Verify the install

```bash
python -c "
from neoforce.visuo_tactile_conversion import TactileConversionModel
TactileConversionModel(device='cpu')
print('tactile conversion model OK')
"

python -c "
import torch
from neoforce.models import NeoForceConfig, NeoForceEncoder
ck = torch.load('weights/neoforce.pt', map_location='cpu')
known = NeoForceConfig.__dataclass_fields__
cfg = NeoForceConfig(**{k: v for k, v in ck['model_cfg'].items() if k in known})
model = NeoForceEncoder(cfg)
result = model.load_state_dict(ck['model'], strict=False)
assert not result.missing_keys
print('NeoForce OK, step', ck['step'],
      '|', sum(p.numel() for p in model.parameters())/1e6, 'M params')
out = model(torch.randn(1, 2, 3, *cfg.image_size))
print('force', tuple(out['force'].shape), 'mask', tuple(out['mask_logits'].shape))
"
```

Force and mask both come back at `(1, 2, 6, 360, 640)`, the full input resolution.