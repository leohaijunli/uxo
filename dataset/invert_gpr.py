"""
Invert the simulated GPR B-scans to estimate the position of each UXO.

A buried target produces a diffraction hyperbola in the B-scan. From it we
estimate, per object:
  - cross-range position  xc   (apex trace x)              -> true xc = 0.30 m
  - depth to target TOP        (from 2-way apex travel time)
  - soil velocity              (hyperbola curvature fit)   -> true v = 0.122 m/ns
and compare against the ground truth / simulation geometry.

Antenna is flown air_height above the surface, so the 2-way apex time is
    t_apex = 2*air_h/c_air + 2*depth_top/v_soil
hence   depth_top = (t_apex - 2*air_h/c_air) * v_soil / 2.

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/invert_gpr.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G3D = os.path.join(HERE, "gpr3d")

C0 = 0.299792458            # m/ns
ER = 6.0
V_SOIL = C0 / np.sqrt(ER)   # 0.1224 m/ns
AIR_H = 0.30                # antenna height above surface (m)
T_AIR = 2 * AIR_H / C0      # 2-way air-gap delay (ns) ~2.0
XC_TRUE = 0.30


from scipy.signal import hilbert


def envelope(Ez):
    return np.abs(hilbert(Ez, axis=-1))


def hyperbola_velocity(x, t, A, apex_t, apex_x, window=0.10):
    """Fit soil velocity from the apex-arrival times of nearby traces."""
    r = np.abs(x - apex_x)
    w = r < window
    if w.sum() < 3:
        return float("nan")
    tarr = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        tarr[i] = t[np.argmax(A[i])]
    m, _b = np.polyfit(r[w] ** 2, (tarr[w] - apex_t) ** 2, 1)
    return float(np.sqrt(1.0 / m))


def invert_one(path):
    d = np.load(path)
    Ez, x, t = d["Ez"], d["x"], d["t"]
    if Ez.ndim == 3:
        Ez = Ez[:, 0, :]
    t = np.asarray(t, dtype=float) * 1e9          # convert s -> ns
    meta = dict(ID=str(d["ID"]), item=str(d["item"]),
                material=str(d["material"]), radius=float(d["radius_m"]),
                depth_meta=float(d["depth_top_m"]))
    A = envelope(Ez)
    w = t > 2.2                                   # skip direct-wave residue (t<~2 ns)
    ap_tr = int(np.argmax(A[:, w].max(axis=1)))
    apex_x = float(x[ap_tr])
    apex_t = float(t[w][np.argmax(A[ap_tr, w])])
    v_fit = hyperbola_velocity(x, t, A, apex_t, apex_x)
    # use fitted velocity if it is plausible, else the known soil value
    v = v_fit if 0.05 < v_fit < 0.25 else V_SOIL
    depth_top = (apex_t - T_AIR) * v / 2.0          # depth to target TOP (m)
    depth_cent = depth_top + meta["radius"]         # depth to centre
    return {**meta,
            "apex_x": apex_x, "dx_cm": (apex_x - XC_TRUE) * 100.0,
            "apex_t_ns": apex_t, "t_air_ns": T_AIR,
            "v_fit": v_fit, "v_used": v,
            "depth_top_est_cm": depth_top * 100.0,
            "depth_cent_est_cm": depth_cent * 100.0,
            "sim_depth_top_cm": min(meta["depth_meta"], 0.15) * 100.0,
            "err_top_cm": (depth_top - min(meta["depth_meta"], 0.15)) * 100.0}


def main():
    rows = [invert_one(f)
            for f in sorted(glob.glob(os.path.join(G3D, "*_bscan.npz")))]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== CROSS-RANGE position estimate (true xc = %.2f m) ===" % XC_TRUE)
    print(df[["ID", "item", "material", "apex_x", "dx_cm"]]
          .round(2).to_string(index=False))
    print("\n=== DEPTH estimate (target TOP below surface) ===")
    print("air delay %.2f ns + soil 2-way (v=%.3f m/ns)" % (T_AIR, V_SOIL))
    print(df[["ID", "depth_top_est_cm", "sim_depth_top_cm", "err_top_cm"]]
          .round(2).to_string(index=False))
    print("\n=== fitted soil velocity (expected %.3f m/ns) ===" % V_SOIL)
    print(df[["ID", "v_fit"]].round(3).to_string(index=False))
    # summary stats
    print("\n=== summary ===")
    print("cross-range |dx| mean = %.2f cm" % df["dx_cm"].abs().mean())
    print("depth error  mean = %.2f cm, rms = %.2f cm"
          % (df["err_top_cm"].mean(), np.sqrt(np.mean(df["err_top_cm"] ** 2))))


if __name__ == "__main__":
    main()