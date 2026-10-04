"""
Full 3-D Kirchhoff (diffraction-stack) migration inversion of a GPR grid cube.

Given a background-subtracted data cube Ez[nx, ny, nt] over an antenna grid
(X, Y), migrate it into a 3-D image volume I(x, y, z) and locate the target
as the image peak. Returns the full 3-D position (x, y, depth) and compares
with the known target position.

Two-way travel time for a buried point (x,y,z) under a common-offset antenna
at (xs, ys) with receiver at (xs+rx, ys), through an air layer (height h, speed
c0) then soil (speed v):

    t = (R_air_src + R_air_rx) / c0 + (R_soil_src + R_soil_rx) / v

For a shallow ground-coupled survey we approximate the air layer as a fixed
two-way delay + a soil-only ray path through an effective single medium:

    t = 2*h/c0 + 2*sqrt((x-xs)^2 + (y-ys)^2 + z^2) / v

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/invert_gpr3d.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
G3D = os.path.join(HERE, "gpr3d_grid")

C0 = 0.299792458            # m/ns
ER = 6.0
V_SOIL = C0 / np.sqrt(ER)   # 0.1224 m/ns
AIR_H = 0.05                # antenna height (ground-coupled)
T_AIR = 2 * AIR_H / C0      # ns

NZ = 41                     # migration depth samples
ZRANGE = (0.0, 0.30)        # m, depth window


def migrate(Ez, X, Y, t, xc, yc, rx=0.06, v=V_SOIL, dz=ZRANGE, dec=1):
    """Diffraction-stack migration. Returns (I, xg, yg, zg)."""
    xg = np.linspace(xc - 0.10, xc + 0.10, X.size * 2 // dec)
    yg = np.linspace(yc - 0.10, yc + 0.10, Y.size * 2 // dec)
    zg = np.linspace(dz[0], dz[1], NZ // dec)
    A = np.abs(hilbert(Ez, axis=-1))
    I = np.zeros((xg.size, yg.size, zg.size))
    for i, xi in enumerate(xg):
        for j, yi in enumerate(yg):
            for k, zi in enumerate(zg):
                tt = (2 * np.sqrt((X - xi) ** 2 + (Y - yi) ** 2 + zi ** 2) / v
                      + T_AIR)
                idx = np.clip(np.searchsorted(t, tt.ravel()), 0, t.size - 2)
                frac = (tt.ravel() - t[idx]) / (t[idx + 1] - t[idx])
                val = (A[..., idx] * (1 - frac) + A[..., idx + 1] * frac)
                I[i, j, k] = val.mean()
    return I, xg, yg, zg


def best_velocity(Ez, X, Y, t, xc, yc, vs=(0.085, 0.10, 0.115, 0.122)):
    """Pick the migration velocity that focuses the image best (max peak, min
    background):  focus = peak / mean of the top 5% image values."""
    best = (0, None)
    for v in vs:
        I, xg, yg, zg = migrate(Ez, X, Y, t, xc, yc, v=v, dec=2)
        flat = np.sort(I.ravel())
        top = flat[-max(1, flat.size // 20):]
        focus = flat[-1] / (top.mean() + 1e-12)
        if best[0] < focus:
            best = (focus, v)
    return best[1]


def invert_cube(path):
    d = np.load(path)
    Ez, X, Y, t = d["Ez"], d["X"], d["Y"], d["t"]
    t = t * 1e9
    xc = float(d.get("xc", 0.30)); yc = float(d.get("yc", 0.125))
    meta = dict(ID=str(d["ID"]), item=str(d["item"]),
                material=str(d["material"]), depth=float(d["depth_top_m"]))
    v = best_velocity(Ez, X, Y, t, xc, yc)
    I, xg, yg, zg = migrate(Ez, X, Y, t, xc, yc, v=v)
    pi, pj, pk = np.unravel_index(np.argmax(I), I.shape)
    return {**meta, "v_best": v,
            "x_est": xg[pi], "y_est": yg[pj], "z_est": zg[pk],
            "x_true": xc, "y_true": yc, "z_true": meta["depth"],
            "dx_cm": (xg[pi] - xc) * 100, "dy_cm": (yg[pj] - yc) * 100,
            "dz_cm": (zg[pk] - meta["depth"]) * 100,
            "peak": float(I[pi, pj, pk])}


def main():
    rows = [invert_cube(f)
            for f in sorted(glob.glob(os.path.join(G3D, "*_cube.npz")))]
    if not rows:
        print("no cubes found in", G3D)
        return
    df = pd.DataFrame(rows)
    print("=== 3-D migration position estimates (true: x=0.30 y=0.125) ===")
    print(df[["ID", "item", "x_est", "y_est", "z_est",
              "dx_cm", "dy_cm", "dz_cm"]].round(2).to_string(index=False))
    print("\n=== summary ===")
    for ax in ("dx_cm", "dy_cm", "dz_cm"):
        print(ax, "mean=%.2f rms=%.2f" % (df[ax].mean(), np.sqrt(np.mean(df[ax] ** 2))))


if __name__ == "__main__":
    main()