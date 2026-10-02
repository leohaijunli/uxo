"""
Validate the simulated GPR B-scans are physically reasonable.

Reads dataset/gpr3d/<ID>_bscan.npz (background-subtracted Ez) and checks:
  1. apex (2-way target) arrival time  vs  theoretical 2*depth/v_soil
  2. hyperbola curvature -> fitted soil velocity vs er=6 expectation
  3. peak cross-range position x vs the object xc
  4. peak echo amplitude metal (pec) vs plastic / air  -> contrast
  5. background-subtraction quality (residual direct wave level)

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/check_gpr.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
G3D = os.path.join(HERE, "gpr3d")
G2D = os.path.join(HERE, "gpr2d")

C0    = 0.299792458          # m/ns
ER    = 6.0                  # soil dielectric (handoff section 5e)
V_SOIL = C0 / np.sqrt(ER)    # ~0.1224 m/ns
XC    = 0.30                 # target x-centre (3-D scene)
X2D   = 0.50                 # target x-centre (2-D scene)


def hyperbola_velocity(x, t, apex_t, apex_x, depths=None):
    """Estimate soil velocity from hyperbola curvature near the apex."""
    r = np.sqrt((x - apex_x) ** 2)
    w = r < 0.12                                  # window near apex
    if w.sum() < 3:
        return float("nan")
    dt = t[w] - apex_t
    r = r[w]
    # two-way: t = 2*sqrt((r/2)^2 + d^2)/v  ->  t^2 = r^2/v^2 + const
    m, b = np.polyfit(r ** 2, dt ** 2, 1)
    return float(np.sqrt(1.0 / m))


def check_one(path, xc):
    d = np.load(path)
    Ez, x, t = d["Ez"], d["x"], d["t"]
    meta = dict(ID=str(d["ID"]), item=str(d["item"]),
                material=str(d["material"]), radius=float(d["radius_m"]),
                depth=float(d["depth_top_m"]))
    if Ez.ndim == 3:                               # 3-D dataset has (n,1,n)
        Ez = Ez[:, 0, :]
    ap_x = int(np.argmax(np.abs(Ez).max(axis=1)))
    apex_x = float(x[ap_x])
    gate = (t > 3.0) & (t < 11.0)
    apex_t = float(t[gate][np.argmax(np.abs(Ez[ap_x, gate]))])
    depth = meta["depth"]
    theor = 2.0 * depth / V_SOIL                     # ns, 2-way to target top
    vfit = hyperbola_velocity(x, t, apex_t, apex_x)
    amp = float(np.abs(Ez[ap_x, gate]).max())
    dirlevel = float(np.abs(Ez[:, (t < 2.5)]).max())
    return {**meta,
            "xc": xc, "apex_x": apex_x, "apex_t": apex_t, "theor_t": theor,
            "t_err_ns": apex_t - theor, "v_fit": vfit, "v_expected": V_SOIL,
            "v_err_pct": (vfit - V_SOIL) / V_SOIL * 100.0,
            "amp": amp, "direct_resid": dirlevel,
            "snr": amp / max(dirlevel, 1e-12)}


def main():
    rows = []
    for ddir, xc in ((G3D, XC), (G2D, X2D)):
        for f in sorted(glob.glob(os.path.join(ddir, "*_bscan.npz"))):
            try:
                rows.append(check_one(f, xc))
            except Exception as e:
                rows.append(dict(ID=os.path.basename(f), error=str(e)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No B-scans found. Run the GPR build first.")
        return
    ok = df[df.get("error").isna()]
    print(f"validated {len(ok)} B-scans")
    print("\n-- target arrival time check (apex vs theoretical 2*d/v) --")
    if len(ok):
        print(ok[["ID", "material", "depth", "theor_t", "apex_t", "t_err_ns"]]
              .round(3).to_string(index=False))
    print("\n-- hyperbola velocity fit (expected %.3f m/ns for er=6) --"
          % V_SOIL)
    if len(ok):
        print(ok[["ID", "material", "v_fit", "v_err_pct"]].round(3)
              .to_string(index=False))
    print("\n-- cross-range alignment (xc=%.2f) --" % XC)
    if len(ok):
        print(ok[["ID", "xc", "apex_x"]].round(3).to_string(index=False))
    print("\n-- echo amplitude by material (metal must be >> plastic/air) --")
    if len(ok):
        grp = ok.groupby("material")["amp"].agg(["count", "median", "max"])
        print(grp.round(3).to_string())
    print("\n-- background-subtraction quality (residual direct wave) --")
    if len(ok):
        print(ok[["ID", "snr", "direct_resid"]].round(3)
              .to_string(index=False))


if __name__ == "__main__":
    main()