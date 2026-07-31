"""Build a simplified USB peg mesh (box body + rectangular USB-A connector, connector facing -Z).
Usage: python scripts/asset_tools/build_usb.py [out.obj]   (default /tmp/usb_assets/USB.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usb_assets/USB.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

BODY = [19.9, 9.5, 52.0]      # body width x thickness x length (mm)
CONN = [12.0, 4.5, 14.0]      # connector: standard USB-A 12x4.5, length 14 (~3mm embedded in the body)
body = trimesh.creation.box(extents=BODY); body.apply_translation([0, 0, BODY[2] / 2])
conn = trimesh.creation.box(extents=CONN); conn.apply_translation([0, 0, -CONN[2] / 2 + 3.0])
usb = trimesh.boolean.union([body, conn])         # watertight single body
usb.apply_translation(-usb.bounds.mean(axis=0))   # center at origin (connector end at -Z)
usb.apply_scale(0.001)                            # mm -> m
usb.export(out)
print(f"saved {out}  extents(mm)={np.round(usb.extents*1000,2)}  watertight={usb.is_watertight}")
