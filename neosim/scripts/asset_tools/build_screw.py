"""Build a "screw" (thumbtack-shaped) peg mesh: cylindrical shaft + flat head plate.
Shaft along Z, head at the +Z end; centered at the origin.
Usage: python scripts/asset_tools/build_screw.py [out.obj]   (default /tmp/screw_assets/SCREW.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/screw_assets/SCREW.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

SHAFT_D, SHAFT_H = 20.0, 70.0     # shaft diameter / height (mm)
HEAD_W,  HEAD_H  = 44.0, 8.0      # head (square plate) side / thickness (mm); flat faces give the gripper a stable planar grasp

shaft = trimesh.creation.cylinder(radius=SHAFT_D / 2, height=SHAFT_H, sections=64)
shaft.apply_translation([0, 0, SHAFT_H / 2])                       # shaft bottom at z=0
head = trimesh.creation.box(extents=[HEAD_W, HEAD_W, HEAD_H])      # square head plate
head.apply_translation([0, 0, SHAFT_H + HEAD_H / 2 - 1.5])         # plate seated on shaft top (1.5mm overlap)

screw = trimesh.boolean.union([shaft, head])      # watertight single body
screw.apply_translation(-screw.bounds.mean(axis=0))   # center; head at +Z
screw.apply_scale(0.001)                              # mm -> m
screw.export(out)
print(f"saved {out}  extents(mm)={np.round(screw.extents*1000,2)}  watertight={screw.is_watertight}")
