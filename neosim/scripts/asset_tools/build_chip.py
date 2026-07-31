"""Build a thin chip/strip mesh (slightly arched thin slab).
Usage: python scripts/asset_tools/build_chip.py [out.obj]   (default /tmp/chip_assets/CHIP.obj)
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/chip_assets/CHIP.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

L, W, T = 82.5, 18.0, 2.4       # length x width x thickness (mm), scaled up 1.5x
ARC = 13.5                      # longitudinal arch: ends raised ~13.5mm above the middle

box = trimesh.creation.box(extents=[L, W, T])
box = box.subdivide_to_size(1.8)            # subdivide so the arch is smooth and tets are well-shaped
V = box.vertices.copy()
V[:, 2] += ARC * (V[:, 0] / (L / 2)) ** 2   # slight upward bend along the length
chip = trimesh.Trimesh(V, box.faces, process=True)
chip.merge_vertices()

chip.apply_translation(-chip.bounds.mean(axis=0))   # center
chip.apply_scale(0.001)                             # mm -> m
chip.export(out)
print(f"saved {out}  extents(mm)={np.round(chip.extents*1000,2)}  "
      f"verts={len(chip.vertices)} faces={len(chip.faces)} watertight={chip.is_watertight}")
