"""
gprMax 2-D TM B-scan generator for the OSU seeded UXO targets.

VALIDATED (gprMax 4.0.0, CPU/OpenMP, Python 3.12 in .venv-gpr).

Run:  MPLCONFIGDIR=/tmp/mpl ../.venv-gpr/bin/python gpr_bs_scan.py
"""
import os, numpy as np, h5py, gprMax

DL       = 0.004    # m, mesh (4 mm)
FREQ     = 1.5e9    # Hz, Ricker
TW       = 12e-9    # s

# --- key gprMax-4 conventions (do not "fix" these) -------------------------
#  * DomainMode(mode="TM")            -> NOT "2D TM"
#  * Domain(p1=(Lx, Ly, inf))         -> the INFINITE axis is z (last)
#  * x = scan direction, y = depth, z = target long axis
#  * omit PMLThickness                -> explicit PMLThickness(10) FAILS in 2-D
#  * use builtin material id "pec"    -> declaring se=1e7 triggers a
#                                        dispersion-resolution error
#  * multi-trace: pass a LIST of scenes, n=len(scenes); NEVER pass i=0
#    (i must be >0 or omitted). Output is outfile1.h5, outfile2.h5, ...
# ---------------------------------------------------------------------------


def make_scene(xsrc, xc, radius, depth_top, er_soil=6.0, sm_soil=0.001,
               material_id="pec", er_obj=1.0, se_obj=0.0,
               Lx=1.0, Ld=0.85, air_height=0.30, rx_offset=0.06, nthreads=4,
               add_target=True):
    """One B-scan trace. depth_top = depth to TOP of the target (m).
    air_height = antenna height above the surface (from the real survey).
    Convention: x=scan, y=depth, z=infinite; surface at y=SOIL_TOP (air below,
    soil above), antenna flown air_height above it. Domain is [0,0,0]..[Lx,Ld]."""
    soil_top = air_height + 0.05
    sc = gprMax.Scene()
    sc.add(gprMax.Title("uxo_2d"))
    sc.add(gprMax.Domain(p1=(Lx, Ld, np.inf)))
    sc.add(gprMax.DomainMode(mode="TM"))
    sc.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    sc.add(gprMax.TimeWindow(time=TW))
    sc.add(gprMax.OMPThreads(n=nthreads))
    sc.add(gprMax.Material(er=1.0, se=0, mr=1, sm=0, id="air"))
    sc.add(gprMax.Box(p1=(0, 0, 0), p2=(Lx, Ld, np.inf), material_id="air"))
    sc.add(gprMax.Material(er=er_soil, se=0, mr=1, sm=sm_soil, id="soil"))
    sc.add(gprMax.Box(p1=(0, soil_top, 0), p2=(Lx, Ld, np.inf), material_id="soil"))
    yc = soil_top + depth_top + radius
    tgt = material_id
    if tgt != "pec":
        sc.add(gprMax.Material(er=er_obj, se=se_obj, mr=1, sm=0, id=tgt))
    if add_target:
        sc.add(gprMax.Cylinder(p1=(xc, yc, 0), p2=(xc, yc, np.inf), r=radius,
                               material_id=tgt))
    sc.add(gprMax.Waveform(wave_type="ricker", amp=1.0, freq=FREQ, id="src"))
    sc.add(gprMax.HertzianDipole(p1=(xsrc, soil_top - air_height, np.inf),
                                 polarisation="z", waveform_id="src"))
    sc.add(gprMax.Rx(p1=(xsrc + rx_offset, soil_top - air_height, np.inf),
                     id="rx", outputs=["Ez"]))
    return sc


def run_bscan(outstem, xc, radius, depth_top, x0=None, n=31, dx=0.01,
              add_target=True, air_height=0.30, **kw):
    """Build & run an n-trace common-offset B-scan. Returns (X, B, t)."""
    x0 = xc - (n // 2) * dx if x0 is None else x0
    scenes = [make_scene(x0 + dx * k, xc, radius, depth_top,
                         add_target=add_target, air_height=air_height,
                         **kw) for k in range(n)]
    gprMax.run(scenes=scenes, n=len(scenes), outputfile=outstem,
               hide_progress_bars=True, cpu_precision="single", log_level=40)
    B, X = [], []
    for k in range(1, n + 1):
        with h5py.File(f"{outstem}{k}.h5", "r") as f:
            rx = list(f["rxs"].keys())[0]
            B.append(f["rxs"][rx]["Ez"][:].copy())
            X.append(f["rxs"][rx].attrs["Position"][0])
    B = np.array(B)
    t = np.linspace(0, TW, B.shape[1])
    return np.array(X), B, t


if __name__ == "__main__":
    X, B, t = run_bscan("/tmp/uxo/demo", xc=0.50, radius=0.05, depth_top=0.06, n=31)
    resid = B - B.mean(axis=0, keepdims=True)      # background removal
    gate = (t > 5e-9) & (t < 12e-9)
    amp = np.abs(resid[:, gate]).max(axis=1)
    print("traces:", len(X), " x:", X[0], "->", X[-1])
    print("residual peak amplitude along scan:")
    print(np.round(amp, 2))
    print("peak at x =", X[int(np.argmax(amp))], " (target at 0.50)")
