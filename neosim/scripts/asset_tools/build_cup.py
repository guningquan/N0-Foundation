"""Build a "paper cup" mesh: a frustum wider at the top, thin-walled, closed bottom,
open top, with an outward rolled lip around the rim.
   The lip makes stacked cups wedge together with a small gap, so they resist
   separation (matches the "separate stacked paper cups" task).
   Cup axis along Z, rim at the +Z end; centered at the origin.
Usage: python scripts/asset_tools/build_cup.py [out.obj]   (default /tmp/cup_assets/CUP.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cup_assets/CUP.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- Geometry (mm) ----
R_TOP   = 38.0    # rim outer radius (D76)
R_BASE  = 27.0    # base outer radius (D54); taper (wider at top) allows nesting
H       = 92.0    # cup height
WALL    = 1.6     # wall thickness (thin-walled paper cup)
BOTTOM  = 2.5     # bottom thickness
LIP_OUT = 2.2     # lip outward protrusion
LIP_H   = 4.0     # lip height (measured down from the rim)
SECTIONS = 96     # revolve subdivisions

# Inner radius at the rim (inner edge of the wall top)
r_top_in  = R_TOP - WALL
# Inner radius at the base (bottom of the inner wall)
r_base_in = R_BASE - WALL

# Closed half-section polygon (r, z): up the outer wall -> lip -> down the inner wall -> closed bottom
profile = np.array([
    [0.0,        0.0],          # outer bottom center
    [R_BASE,     0.0],          # outer bottom edge
    [R_TOP,      H - LIP_H],    # outer wall up to the lip underside
    [R_TOP + LIP_OUT, H - LIP_H],  # lip protrusion
    [R_TOP + LIP_OUT, H],       # lip top outer edge
    [r_top_in,   H],            # rim inner edge
    [r_base_in,  BOTTOM],       # inner wall down to the inner bottom
    [0.0,        BOTTOM],       # inner bottom center
])

cup = trimesh.creation.revolve(profile, sections=SECTIONS)   # revolve around Z into a solid cup
cup.apply_translation(-cup.bounds.mean(axis=0))              # center
cup.apply_scale(0.001)                                       # mm -> m
cup.export(out)
print(f"saved {out}  extents(mm)={np.round(cup.extents*1000,2)}  "
      f"verts={len(cup.vertices)} faces={len(cup.faces)} watertight={cup.is_watertight}")
