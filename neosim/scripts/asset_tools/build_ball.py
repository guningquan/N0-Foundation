"""Build a small ball (marble) for the "pour balls from red cup into green cup" task.
   Balls must be small enough to fit in the cup, coexist in numbers, and pour out
   through the rim. Cup rim inner diameter ~D73mm, base inner diameter ~D51mm,
   so use D16mm balls (radius 8mm): several fit in a cup and slide out smoothly.
   An icosphere gives uniform triangles and stable tet quality; centered at the origin.
Usage: python scripts/asset_tools/build_ball.py [out.obj]   (default /tmp/ball_assets/BALL.obj)
env: ~/miniconda3/envs/UniVTAC/bin/python
"""
import sys, os, numpy as np, trimesh

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ball_assets/BALL.obj"
os.makedirs(os.path.dirname(out), exist_ok=True)

R = 8.0          # ball radius (mm); D16mm ball
SUBDIV = 3       # icosphere subdivisions: 642 verts / 1280 faces, smooth surface, uniform tets

ball = trimesh.creation.icosphere(subdivisions=SUBDIV, radius=R)
ball.apply_translation(-ball.bounds.mean(axis=0))   # center at origin
ball.apply_scale(0.001)                             # mm -> m
ball.export(out)
print(f"saved {out}  extents(mm)={np.round(ball.extents*1000,2)}  "
      f"verts={len(ball.vertices)} faces={len(ball.faces)} watertight={ball.is_watertight}")
