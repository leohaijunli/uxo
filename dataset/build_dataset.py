"""
Build a multi-modal (GPR + magnetic) dataset for the simulable UXO objects.

    manifest.csv                one row per object
    mag/<ID>_mag.npz            measured magnetic patch (real survey)
    gpr2d/<ID>_bscan.npz        simulated 2-D GPR B-scan   (fast proxy)
    gpr3d/<ID>_bscan.npz        simulated 3-D GPR B-scan   (faithful)

Stages are independent:
    python build_dataset.py --stage mag
    python build_dataset.py --stage gpr3d --ids A3 B1 F6
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from invert3d import lonlat_to_xy, load_ground_truth, load_magnetometer  # noqa

PLAN = pd.read_csv(os.path.join(ROOT, "report_output_gpr", "gpr_model_plan.csv"))

MAG_CELL   = 0.25     # m, patch cell size
MAG_HALF   = 1.50     # m, patch half-width  -> 13x13 cells centred on the object
MAG_SMOOTH = 0.6      # cells

# property table:  material -> (er, se, metal?, kind)
MAT = {"pec": (1.0, 1e7), "plastic": (2.9, 0.003),
       "explosive": (3.0, 0.002), "air": (1.0, 0.0)}


def simulable():
    return PLAN[PLAN.feasible.isin(["yes", "limited"])].copy()


# --------------------------------------------------------------------------- #
# magnetic                                                                     #
# --------------------------------------------------------------------------- #
def build_mag(ids=None, cell=MAG_CELL, half=MAG_HALF):
    out = os.path.join(HERE, "mag"); os.makedirs(out, exist_ok=True)
    s = load_magnetometer()
    gt = load_ground_truth()
    if ids:
        gt = gt[gt.ID.isin(ids)]

    # grid the whole survey once, for every magnetic channel
    x, y = s.x.to_numpy(), s.y.to_numpy()
    a = s.anomaly.to_numpy()
    comp = {c: s[c].to_numpy() if c in s else None for c in ("TMI", "Bx", "By", "Bz")}
    x0, y0 = np.floor(x.min() / cell) * cell, np.floor(y.min() / cell) * cell
    gx = np.arange(x0, x.max() + cell, cell)
    gy = np.arange(y0, y.max() + cell, cell)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[x, y]

    def grid(v):
        Z = griddata(pts, v, (GX, GY), method="linear")
        n = ~np.isfinite(Z)
        if n.any():
            Z[n] = griddata(pts, v, (GX, GY), method="nearest")[n]
        return gaussian_filter(Z, MAG_SMOOTH)

    A = grid(a)
    B = {c: (grid(v) if v is not None else None) for c, v in comp.items()}
    # signed background level for Bx/By/Bz so the patch stores an *anomaly*
    B0 = {c: np.nanmedian(v) if v is not None else None for c, v in B.items()}

    hw = int(round(half / cell))
    rows = []
    for _, o in gt.iterrows():
        ox, oy = lonlat_to_xy(o.Longitude, o.Latitude)
        ix = int(np.argmin(np.abs(gx - ox))); iy = int(np.argmin(np.abs(gy - oy)))
        sl = (slice(max(iy - hw, 0), iy + hw + 1), slice(max(ix - hw, 0), ix + hw + 1))
        patch = dict(ID=o.ID, gx=gx[sl[1]] - ox, gy=gy[sl[0]] - oy,
                     anomaly=A[sl])
        for c in ("TMI", "Bx", "By", "Bz"):
            if B[c] is not None:
                v = B[c][sl] - (0.0 if c == "TMI" else B0[c])
                patch[c] = v
        np.savez_compressed(os.path.join(out, f"{o.ID}_mag.npz"), **patch)
        rows.append(dict(ID=o.ID, item=o.Item, cls=o.Class,
                         x=ox, y=oy, depth_cm=o.Depth,
                         ferrous=bool(o.ferrous),
                         mag_anom_peak=float(np.nanmax(np.abs(A[sl]))),
                         mag_anom_rms=float(np.sqrt(np.nanmean(A[sl] ** 2)))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "mag_index.csv"), index=False)
    print(f"mag: wrote {len(df)} patches -> {out}")
    print(df.sort_values("mag_anom_peak", ascending=False).head(10).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# GPR (simulated)                                                              #
# --------------------------------------------------------------------------- #
def target_geometry(row):
    """(radius_m, length_m or None, depth_top_m) from the plan row.
    length=None -> compact sphere (also used for air voids); otherwise a finite
    cylinder whose length is clamped to fit the canonical 3-D domain (Ly=0.25).
    """
    D = row.Diameter_cm; L = row.Length_cm
    size = np.nan
    for v in (D, L):
        if pd.notna(v):
            size = np.nanmax([size, v]) if pd.notna(size) else v
    if not np.isfinite(size):
        size = 20.0 if "cache" in str(row.Item).lower() else np.nan
    r = (D / 2.0 if pd.notna(D) else size / 2.0) / 100.0
    depth = (row.depth_cm / 100.0) if pd.notna(row.depth_cm) else 0.05

    if str(row.model) == "sphere":
        length = None
    elif pd.notna(L):
        length = min(L / 100.0, 0.10)          # clamp to canonical 3-D domain
    else:
        length = None
    return float(r), length, float(depth)


def build_gpr(ids, mode="3d", ntraces=21, dx=0.02):
    """mode: '2d' (fast proxy) or '3d' (faithful). dx defaults to the real
    survey along-track spacing (speed/PRF ~0.02 m).

    Runs a SINGLE canonical no-target background model (shared by every object)
    and stores the background-subtracted field  Ez = Ez_target - Ez_background,
    so the direct Tx-Rx wave is removed (see HANDOFF section 6c).
    """
    out = os.path.join(HERE, f"gpr{mode}"); os.makedirs(out, exist_ok=True)
    plan = PLAN.set_index("ID")

    # --- one shared background run (no target) -----------------------------
    bgstem = os.path.join("/tmp/uxo", f"bg_{mode}")
    if mode == "3d":
        from gpr_sim.gpr_bs_scan_3d import run_bscan_3d
        _, Bg, tg = run_bscan_3d(bgstem, xc=0.30, r=0.05, depth_top=0.05,
                                 n=ntraces, dx=dx, add_target=False)
        run_target = lambda st, r, depth, length: \
            run_bscan_3d(st, xc=0.30, r=r, depth_top=min(depth, 0.15),
                         length=length, n=ntraces, dx=dx)
    else:
        from gpr_sim.gpr_bs_scan import run_bscan
        _, Bg, tg = run_bscan(bgstem, xc=0.50, radius=0.05, depth_top=0.05,
                              n=ntraces, dx=dx, add_target=False)
        run_target = lambda st, r, depth, length: \
            run_bscan(st, xc=0.50, radius=r, depth_top=min(depth, 0.15),
                      n=ntraces, dx=dx)

    rows = []
    for i, idv in enumerate(ids, 1):
        row = plan.loc[idv]
        r, length, depth = target_geometry(row)
        er, se = MAT[row.material]
        try:
            X, B, t = run_target(os.path.join("/tmp/uxo", f"ds_{idv}"),
                                 r, depth, length)
            Ez = B - Bg
        except Exception as e:                                    # noqa
            print(f"  !! {idv}: {e}"); rows.append(dict(ID=idv, ok=False)); continue
        np.savez_compressed(os.path.join(out, f"{idv}_bscan.npz"),
                            x=X, t=t, Ez=Ez, ID=idv, item=str(row.Item),
                            material=row.material, radius_m=r,
                            length_m=length, depth_top_m=depth, mode=mode,
                            bg_subtracted=True)
        rows.append(dict(ID=idv, ok=True, item=str(row.Item),
                         material=row.material, radius_m=r,
                         length_m=length, depth_top_m=depth, ntraces=len(X)))
        print(f"  [{i}/{len(ids)}] {idv} {str(row.Item)[:22]:22s} r={r:.3f} "
              f"d={depth:.2f} L={length if length is None else round(length,2)} "
              f"-> {out}/{idv}_bscan.npz")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "gpr_index.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="mag", choices=["mag", "gpr2d", "gpr3d", "manifest"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--ntraces", type=int, default=21)
    ap.add_argument("--dx", type=float, default=0.01)
    A = ap.parse_args()
    ids = A.ids or simulable().ID.tolist()
    if A.stage == "mag":
        build_mag(ids)
    elif A.stage in ("gpr2d", "gpr3d"):
        build_gpr(ids, mode=A.stage[-2:], ntraces=A.ntraces, dx=A.dx)
