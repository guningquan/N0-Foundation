"""Build the "sleeve": square outside, round through-hole inside (flat outer faces give
the gripper a stable planar grasp; the round hole slides over the screw shaft).
Hole axis along Z, centered.
Usage: python scripts/asset_tools/build_sleeve.py [out.obj]   (default /tmp/screw_assets/SLEEVE.obj)
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/screw_assets/SLEEVE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

SHAFT_D = 20.0                 # mating shaft diameter (mm), matches build_screw.py
CLR = 4.0                      # per-side clearance (mm): widened to 4mm to absorb ~2-3mm lateral alignment error
INNER_D = SHAFT_D + 2 * CLR    # inner hole diameter 28 (round)
OUTER_W = 36.0                 # square outer side length (mm); flat faces for gripper grasp
H = 44.0                       # sleeve length (through-hole, open both ends)

outer = trimesh.creation.box(extents=[OUTER_W, OUTER_W, H])         # square outer wall
inner = trimesh.creation.cylinder(radius=INNER_D / 2, height=H * 1.1, sections=64)  # round through-hole (slightly longer to cut through)
sleeve = trimesh.boolean.difference([outer, inner])   # square outside, round inside

sleeve.apply_translation(-sleeve.bounds.mean(axis=0))  # center
sleeve.apply_scale(0.001)                              # mm -> m
sleeve.export(out)
print(f"saved {out}  outer(square){OUTER_W}mm / inner(round){INNER_D}mm  H={H}  watertight={sleeve.is_watertight}")
