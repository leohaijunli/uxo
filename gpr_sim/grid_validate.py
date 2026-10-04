"""
Validate the 3-D grid scan + full 3-D Kirchhoff migration inversion.

Runs a small nx x ny antenna grid over one object (A1, big metal pressure
cooker) at 5 cm ground-coupled height, then migrates the cube into a 3-D
image volume and locates the target peak (x, y, depth) vs the true position.

Run:  MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python gpr_sim/grid_validate.py
"""
import os, sys, time, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpr_bs_scan_3d as g3

XC, YC = 0.30, 0.125
NX, NY, DX, DY = 7, 7, 0.02, 0.02
R, DEPTH_TOP = 0.115, 0.08          # A1 pressure cooker (metal, pec)

BGDIR = "/tmp/uxo/grid"

if __name__ == "__main__":
    os.makedirs(BGDIR, exist_ok=True)

    def mk_tgt(xs, ys):
        return g3.make_scene(xs, ys, XC, YC, R, depth_top=DEPTH_TOP,
                             material_id="pec")

    def mk_bg(xs, ys):
        return g3.make_scene(xs, ys, XC, YC, R, depth_top=DEPTH_TOP,
                             material_id="pec", add_target=False)

    t0 = time.time()
    print("running background grid ...", flush=True)
    X, Y, Bg, t = g3.run_grid_3d(f"{BGDIR}/bg", NX, NY, DX, DY, XC, YC, mk_bg)
    print(f"background grid done in {time.time()-t0:.0f} s", flush=True)

    t0 = time.time()
    print("running target grid ...", flush=True)
    X, Y, Bt, t = g3.run_grid_3d(f"{BGDIR}/A1", NX, NY, DX, DY, XC, YC, mk_tgt)
    print(f"target grid done in {time.time()-t0:.0f} s", flush=True)

    Ez = Bt - Bg                        # background-subtracted cube (nx,ny,nt)
    np.savez_compressed(f"{BGDIR}/A1_cube.npz", X=X, Y=Y, t=t, Ez=Ez)
    print("cube saved", Ez.shape, "peak sub", np.abs(Ez).max(), flush=True)