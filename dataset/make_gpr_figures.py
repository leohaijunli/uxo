"""
Generate figures for the 3-D grid GPR dataset (run after the cubes exist).

Per object (dataset/gpr3d_grid/<ID>_cube.npz):
  fig<ID>_a  B-scan slice through the target crossline (y=y_centre)
  fig<ID>_b  3-D migrated image: x-y map at the peak depth + x-z & y-z slices
  fig<ID>_c  A-scan over the target + magnetic patch beside it
Summary:
  fig_sum_bscans   all 8 background-subtracted B-scans
  fig_sum_position estimated vs true position (x,y,depth)

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/make_gpr_figures.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G3D = os.path.join(HERE, "gpr3d_grid")
MAG = os.path.join(HERE, "mag")
OUT = os.path.join(ROOT, "report_output_gpr", "figures_gpr3d")
sys.path.insert(0, HERE)
from invert_gpr3d import migrate, best_velocity   # noqa

os.makedirs(OUT, exist_ok=True)


def load_cube(path):
    d = np.load(path)
    Ez = d["Ez"]; X, Y, t = d["X"], d["Y"], d["t"] * 1e9
    return dict(Ez=Ez, X=X, Y=Y, t=t, ID=str(d["ID"]), item=str(d["item"]),
                material=str(d["material"]), depth=float(d["depth_top_m"]),
                xc=float(d.get("xc", 0.30)), yc=float(d.get("yc", 0.125)))


def img(ax, X, Y, C, cmap, vmin=None, vmax=None):
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   origin="lower", aspect="auto")
    return im


def fig_bs(meta):
    Ez, X, Y, t = meta["Ez"], meta["X"], meta["Y"], meta["t"]
    iy = int(np.argmin(np.abs(Y - meta["yc"])))
    fig, ax = plt.subplots(figsize=(6, 4))
    im = img(ax, X, t, Ez[:, iy, :], "RdBu_r", -np.abs(Ez).max(), np.abs(Ez).max())
    ax.axvline(meta["xc"], color="k", lw=0.6, ls=":")
    ax.set(xlabel="x (m)", ylabel="two-way time (ns)", title=f"{meta['ID']} B-scan (y=y_c)")
    fig.colorbar(im, ax=ax, label="Ez")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig{meta['ID']}_a_bscan.png", dpi=110)
    plt.close(fig)


def fig_image(meta):
    Ez, X, Y, t = meta["Ez"], meta["X"], meta["Y"], meta["t"]
    v = best_velocity(Ez, X, Y, t, meta["xc"], meta["yc"])
    I, xg, yg, zg = migrate(Ez, X, Y, t, meta["xc"], meta["yc"], v=v)
    pi, pj, pk = np.unravel_index(np.argmax(I), I.shape)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    im0 = img(ax[0], xg, yg, I[:, :, pk], "hot")
    ax[0].set(xlabel="x (m)", ylabel="y (m)", title=f"x-y at z={zg[pk]:.2f} m (v={v:.3f})")
    im1 = img(ax[1], xg, zg, I[:, pj, :].T, "hot")
    ax[1].set(xlabel="x (m)", ylabel="z (m)", title="x-z slice")
    im2 = img(ax[2], yg, zg, I[pi, :, :].T, "hot")
    ax[2].set(xlabel="y (m)", ylabel="z (m)", title="y-z slice")
    ax[0].plot(meta["xc"], meta["yc"], "c+", ms=12, mew=2)
    fig.colorbar(im0, ax=ax[0]); fig.colorbar(im1, ax=ax[1]); fig.colorbar(im2, ax=ax[2])
    fig.suptitle(f"{meta['ID']} migrated image (peak {xg[pi]:.2f},{yg[pj]:.2f},{zg[pk]:.2f})")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig{meta['ID']}_b_image.png", dpi=110)
    plt.close(fig)


def fig_ascan_mag(meta):
    Ez, X, Y, t = meta["Ez"], meta["X"], meta["Y"], meta["t"]
    ix = int(np.argmin(np.abs(X - meta["xc"]))); iy = int(np.argmin(np.abs(Y - meta["yc"])))
    A = np.abs(hilbert(Ez[ix, iy, :]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t, Ez[ix, iy, :], lw=0.8, label="Ez")
    ax[0].plot(t, A, lw=1.5, label="envelope")
    ax[0].set(xlabel="t (ns)", title=f"A-scan at target ({meta['ID']})"); ax[0].legend()
    mp = os.path.join(MAG, f"{meta['ID']}_mag.npz")
    if os.path.exists(mp):
        m = np.load(mp)
        im = img(ax[1], m["gx"], m["gy"], m["anomaly"], "RdBu_r")
        ax[1].set(xlabel="gx (m)", ylabel="gy (m)", title="magnetic anomaly (real survey)")
        ax[1].set_aspect("equal")
        fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(f"{OUT}/fig{meta['ID']}_c_ascan_mag.png", dpi=110)
    plt.close(fig)


def fig_summary(cubes):
    n = len(cubes)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for k, meta in enumerate(cubes):
        Ez, X, Y, t = meta["Ez"], meta["X"], meta["Y"], meta["t"]
        iy = int(np.argmin(np.abs(Y - meta["yc"])))
        ax = axes.flat[k]
        img(ax, X, t, Ez[:, iy, :], "RdBu_r", -np.abs(Ez).max(), np.abs(Ez).max())
        ax.set_title(f"{meta['ID']} {meta['material']}", fontsize=9)
        ax.set_xlabel("x (m)")
    fig.suptitle("Background-subtracted B-scans (crossline through target)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sum_bscans.png", dpi=110)
    plt.close(fig)


def fig_position(rows):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, col, lab, tru in zip(ax, ("dx_cm", "dy_cm", "dz_cm"),
                                ("x err (cm)", "y err (cm)", "depth err (cm)"),
                                (0, 0, 0)):
        a.bar(df["ID"], df[col]); a.axhline(0, color="k", lw=0.6)
        a.set(title=f"{lab} (RMS {np.sqrt(np.mean(df[col]**2)):.1f})")
        a.tick_params(axis="x", labelrotation=90)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sum_position.png", dpi=110)
    plt.close(fig)


def main():
    files = sorted(glob.glob(os.path.join(G3D, "*_cube.npz")))
    if not files:
        print("no cubes in", G3D); return
    cubes = [load_cube(f) for f in files]
    for m in cubes:
        fig_bs(m); fig_image(m); fig_ascan_mag(m)
        print("figs for", m["ID"], flush=True)
    fig_summary(cubes)
    rows = []
    for m in cubes:
        v = best_velocity(m["Ez"], m["X"], m["Y"], m["t"], m["xc"], m["yc"])
        I, xg, yg, zg = migrate(m["Ez"], m["X"], m["Y"], m["t"], m["xc"], m["yc"], v=v)
        pi, pj, pk = np.unravel_index(np.argmax(I), I.shape)
        rows.append(dict(ID=m["ID"], x_est=xg[pi], y_est=yg[pj], z_est=zg[pk],
                         dx_cm=(xg[pi]-m["xc"])*100, dy_cm=(yg[pj]-m["yc"])*100,
                         dz_cm=(zg[pk]-m["depth"])*100, v_best=v))
    fig_position(rows)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()