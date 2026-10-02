"""
Add real-world spatial sync + drone attitude to every GPR B-scan.

For each simulable object the script writes into dataset/gpr3d/<ID>_bscan.npz
(and gpr2d) and the matching gpr_index.csv:

    x_world, y_world      real survey coordinate of the object (m, from lonlat_to_xy)
    azimuth_deg           local drone flight heading over the object (deg, circular mean)
    pitch_deg, roll_deg   local drone pitch/roll over the object (deg)

This lets a GPR trace be overlaid on the magnetic map at the object's true
position, with the scan-line direction (azimuth) and antenna attitude (pitch,
roll) recorded.

Run:  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/add_world_coords.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from invert3d import lonlat_to_xy, load_ground_truth  # noqa

RADIUS = 3.0          # m, initial search radius for local attitude


def circ_mean(deg):
    a = np.radians(deg[~np.isnan(deg)])
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def local_attitude(ox, oy, px, py, heading, pitch, roll):
    d = np.hypot(px - ox, py - oy)
    for R in (RADIUS, RADIUS * 2, RADIUS * 4, np.inf):
        m = d <= R
        if m.sum() >= 5:
            break
    return (circ_mean(heading[m]), float(np.nanmean(pitch[m])),
            float(np.nanmean(roll[m])), int(m.sum()))


def enrich(ddir):
    pos = pd.read_csv(os.path.join(ROOT, "2024-05-09-18-03-43-position.csv"))
    lat = pd.to_numeric(pos["Latitude RTK"], errors="coerce")
    lon = pd.to_numeric(pos["Longitude RTK"], errors="coerce")
    heading = pd.to_numeric(pos["Heading"], errors="coerce").to_numpy()
    pitch = pd.to_numeric(pos["Pitch"], errors="coerce").to_numpy()
    roll = pd.to_numeric(pos["Roll"], errors="coerce").to_numpy()
    ok = lat.notna() & lon.notna()
    px = np.empty(len(pos)); py = np.empty(len(pos)); px[:] = np.nan; py[:] = np.nan
    for i in np.where(ok)[0]:
        px[i], py[i] = lonlat_to_xy(lon[i], lat[i])

    gt = load_ground_truth()
    idx = os.path.join(ddir, "gpr_index.csv")
    df = pd.read_csv(idx) if os.path.exists(idx) else pd.DataFrame()

    updated = 0
    for f in sorted(glob.glob(os.path.join(ddir, "*_bscan.npz"))):
        idv = os.path.basename(f).replace("_bscan.npz", "")
        o = gt[gt.ID == idv]
        if o.empty:
            continue
        o = o.iloc[0]
        ox, oy = lonlat_to_xy(o.Longitude, o.Latitude)
        az, pit, rl, n = local_attitude(ox, oy, px, py, heading, pitch, roll)
        d = np.load(f)
        keep = {k: d[k] for k in d.files}
        keep.update(dict(x_world=ox, y_world=oy, azimuth_deg=az,
                         pitch_deg=pit, roll_deg=rl, n_attitude=n))
        np.savez_compressed(f, **keep)
        if not df.empty and "ID" in df:
            df.loc[df.ID == idv, ["x_world", "y_world", "azimuth_deg",
                                  "pitch_deg", "roll_deg"]] = [ox, oy, az, pit, rl]
        updated += 1
        print(f"  {idv}: world=({ox:.2f},{oy:.2f}) az={az:.1f} "
              f"pitch={pit:.1f} roll={rl:.1f} (n={n})")
    if not df.empty and "x_world" not in df:
        df["x_world"] = np.nan; df["y_world"] = np.nan
        df["azimuth_deg"] = np.nan; df["pitch_deg"] = np.nan; df["roll_deg"] = np.nan
        for i, r in df.iterrows():
            if os.path.exists(os.path.join(ddir, f"{r.ID}_bscan.npz")):
                d = np.load(os.path.join(ddir, f"{r.ID}_bscan.npz"))
                df.loc[i, ["x_world", "y_world", "azimuth_deg", "pitch_deg",
                           "roll_deg"]] = [d["x_world"], d["y_world"],
                                           d["azimuth_deg"], d["pitch_deg"],
                                           d["roll_deg"]]
    df.to_csv(idx, index=False)
    print(f"{ddir}: enriched {updated} B-scans + index")
    return df


if __name__ == "__main__":
    for ddir in (os.path.join(HERE, "gpr3d"), os.path.join(HERE, "gpr2d")):
        if os.path.isdir(ddir):
            enrich(ddir)