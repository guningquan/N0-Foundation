"""Offline-bake a tetrahedral mesh (fTetWild) for a clean surface mesh, saved as npz.
Usage: python scripts/asset_tools/bake_tet.py <in.obj> <out.npz> [edge_length_r] [epsilon]
  edge_length_r controls tet count (= simulation cost; smaller = denser);
  epsilon controls surface fit/smoothness (smaller = tighter/flatter).
Note: run each asset in its own process (multiple tetrahedralize calls in one process double-free and crash).
"""
import sys, numpy as np, trimesh, wildmeshing as wm

obj, npz = sys.argv[1], sys.argv[2]
elr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0263   # ~1.5mm edge length (for a ~67mm object)
eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001    # tight envelope -> smooth surface

m = trimesh.load(obj, process=False)
if isinstance(m, trimesh.Scene):
    m = m.to_geometry()
pts = np.asarray(m.vertices, float)
tri = np.asarray(m.faces, np.uint32).reshape(-1, 3)

t = wm.Tetrahedralizer(stop_quality=10, max_its=40, epsilon=eps,
                       edge_length_r=elr, skip_simplify=False, coarsen=True)
t.set_log_level(6)
t.set_mesh(pts, tri)
t.tetrahedralize()
tp, ti, _ = t.get_tet_mesh(correct_surface_orientation=True, manifold_surface=True,
                           use_input_for_wn=False)
sp, si = t.get_tracked_surfaces(); sp = sp[0]

np.savez(npz,
         tet_points=np.asarray(tp, np.float32),
         tet_indices=np.asarray(ti).reshape(-1).astype(np.uint32),
         surf_points=np.asarray(sp, np.float32),
         surf_indices=np.asarray(si).reshape(-1).astype(np.uint32))
print(f"{obj} -> {npz}: {len(tp)} pts / {len(np.asarray(ti).reshape(-1))//4} tets")
