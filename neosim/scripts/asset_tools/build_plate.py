"""Build a wider, lower-walled bowl-like tray (a shallow wide bowl) for the dual-arm
plate-stacking task (dual_plate_place_stack).
   Same design as build_bowl.py: the upper half of the outer wall is fully vertical
   (gives the gripper a vertical plane, so it can grasp without welding), and the
   wall blends into the base with a smooth fillet (no sharp corner) so a second
   plate can nest inside. Wider than the bowl (D190) with a lower wall (H30).
   Note: feed only to dual_plate_place_stack (baked into PLATE_STACK.usd); do NOT
   overwrite assets/objects/PLATE.usd -- that shallow plate is shared by
   pour_ball / dual_bowl_unstack / dual_cup_handover_place.
Usage: python scripts/asset_tools/build_plate.py [out.obj]   (default /tmp/plate_assets/PLATE.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/plate_assets/PLATE.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

# ---- Geometry (mm) ----
H = 40.0            # wall height (lower than the bowl's 58mm, but taller than a flat plate to leave room for grasping)
R_OUT = 95.0        # outer radius (D190, larger than the bowl's D130)
WALL = 8.0          # thickened wall (4.5->8mm): a wide thin plate flexes under the gripper; thicker wall stays rigid
BASE_THICK = 8.0
SECTIONS = 160

ZB = -H / 2.0
ZT = H / 2.0
R_IN = R_OUT - WALL

# Low vertical-wall design: upper outer wall at r=R_OUT is fully vertical (grasp band);
# wall-to-base transition is a fillet, with a narrower foot ring for nesting.
R_FOOT = 65.0        # outer radius of the bottom foot ring (fillet start)
Z_KNEE = ZB + 12.0   # height where the outer fillet reaches full radius; vertical above
outer_fillet = np.array([
    [R_FOOT,        ZB],
    [R_FOOT + 9.0,  ZB + 1.5],
    [R_FOOT + 18.0, ZB + 4.0],
    [R_OUT - 4.0,   ZB + 7.0],
    [R_OUT - 1.0,   ZB + 9.0],
    [R_OUT,         Z_KNEE],
], dtype=np.float64)
inner_fillet = np.array([
    [R_IN,                Z_KNEE + 2.0],
    [R_IN - 2.0,          ZB + 8.0],
    [R_IN - 10.0,         ZB + BASE_THICK + 3.0],
    [R_FOOT - WALL - 2.0, ZB + BASE_THICK],
    [0.0,                 ZB + BASE_THICK],   # close the cavity at the well bottom
], dtype=np.float64)
profile = np.vstack([
    [[0.0, ZB]],          # outer bottom center
    outer_fillet,         # fillet foot ring -> full radius
    [[R_OUT, ZT]],        # vertical outer wall (grasp band)
    [[R_IN, ZT]],         # inner rim
    inner_fillet,         # vertical inner wall -> fillet -> well bottom
]).astype(np.float64)

plate = trimesh.creation.revolve(profile, sections=SECTIONS)   # revolve around Z into a solid
plate.apply_translation(-plate.bounds.mean(axis=0))            # center
plate.apply_scale(0.001)                                       # mm -> m
plate.export(out)
print(f"saved {out}  extents(mm)={np.round(plate.extents*1000,2)}  "
      f"verts={len(plate.vertices)} faces={len(plate.faces)} watertight={plate.is_watertight}")
