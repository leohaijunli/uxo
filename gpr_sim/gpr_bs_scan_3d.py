"""
gprMax TRUE 3-D B-scan for the OSU seeded UXO targets.

In 3-D the target is FINITE in every direction (a real object), so the echo is
a genuine point-scatterer hyperbola with correct amplitudes and cross-range
information -- unlike the 2-D TM proxy which assumes an infinitely long
cylinder.

VALIDATED: gprMax 4.0.0 CPU/OpenMP, ~5-7 s per trace at 5 mm mesh.
Run:  MPLCONFIGDIR=/tmp/mpl ../.venv-gpr/bin/python gpr_bs_scan_3d.py
"""
import numpy as np, h5py, gprMax

DL       = 0.005                # m, 5 mm -> ~-2.8% phase-velocity error
FREQ     = 1.5e9
TW       = 12e-9


def make_scene(xs, xc, yc, r, length=None, depth_top=0.06, y0=None,
               er_soil=6.0, sm_soil=0.001, material_id="pec",
               er_obj=1.0, se_obj=0.0,
               Lx=0.60, Ly=0.25, Lz=0.70, soil_depth=0.35, air_height=0.30,
               rx_offset=0.06, nthreads=4, add_target=True):
    """One 3-D trace.  length=None -> sphere (compact); else a FINITE cylinder
    of that length oriented along y (crossline).  depth_top = depth to the top
    of the target in metres.  air_height = antenna height above the surface
    (from the real survey's flight altitude).  add_target=False -> background
    (empty soil) model, shared/canonicalised across every object.

    Domain is [0,0,0]..[Lx,Ly,Lz] (gprMax Domain only takes p1). The ground
    surface is the plane z=SOIL_TOP where SOIL_TOP = air_height + 0.05; air
    occupies z<SOIL_TOP, soil z>SOIL_TOP. The antenna is flown air_height above
    the surface, i.e. at z = SOIL_TOP - air_height."""
    y0 = yc if y0 is None else y0
    soil_top = air_height + 0.05                # surface plane; air below it
    sc = gprMax.Scene()
    sc.add(gprMax.Title("uxo_3d"))
    sc.add(gprMax.Domain(p1=(Lx, Ly, Lz)))
    sc.add(gprMax.DomainMode(mode="3D"))
    sc.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    sc.add(gprMax.TimeWindow(time=TW))
    sc.add(gprMax.OMPThreads(n=nthreads))
    sc.add(gprMax.Material(er=1.0, se=0, mr=1, sm=0, id="air"))
    sc.add(gprMax.Box(p1=(0, 0, 0), p2=(Lx, Ly, Lz), material_id="air"))
    sc.add(gprMax.Material(er=er_soil, se=0, mr=1, sm=sm_soil, id="soil"))
    sc.add(gprMax.Box(p1=(0, 0, soil_top), p2=(Lx, Ly, Lz), material_id="soil"))
    tgt = material_id
    if tgt != "pec":
        sc.add(gprMax.Material(er=er_obj, se=se_obj, mr=1, sm=0, id=tgt))
    if add_target:
        zc = soil_top + depth_top + r           # depth below the surface
        if length:                               # finite rod along y
            sc.add(gprMax.Cylinder(p1=(xc, yc - length / 2, zc),
                                   p2=(xc, yc + length / 2, zc),
                                   r=r, material_id=tgt))
        else:
            sc.add(gprMax.Sphere(p1=(xc, yc, zc), r=r, material_id=tgt))
    sc.add(gprMax.Waveform(wave_type="ricker", amp=1.0, freq=FREQ, id="src"))
    sc.add(gprMax.HertzianDipole(p1=(xs, y0, soil_top - air_height),
                                 polarisation="z", waveform_id="src"))
    sc.add(gprMax.Rx(p1=(xs + rx_offset, y0, soil_top - air_height),
                     id="rx", outputs=["Ez"]))
    return sc


def run_bscan_3d(outstem, xc, r, depth_top, length=None, yc=0.125,
                 n=11, dx=0.02, add_target=True, air_height=0.30, **kw):
    """n-trace 3-D B-scan along x. Returns (X, B, t)."""
    x0 = xc - (n // 2) * dx
    scenes = [make_scene(x0 + dx * k, xc, yc, r, length=length,
                         depth_top=depth_top, add_target=add_target,
                         air_height=air_height, **kw) for k in range(n)]
    gprMax.run(scenes=scenes, n=len(scenes), outputfile=outstem,
               hide_progress_bars=True, cpu_precision="single", log_level=40)
    B, X = [], []
    for k in range(1, n + 1):
        with h5py.File(f"{outstem}{k}.h5", "r") as f:
            rx = list(f["rxs"].keys())[0]
            B.append(f["rxs"][rx]["Ez"][:].copy())
            X.append(f["rxs"][rx].attrs["Position"][0])
    B = np.array(B)
    return np.array(X), B, np.linspace(0, TW, B.shape[1])


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, B, t = run_bscan_3d("/tmp/uxo/d3", xc=0.30, r=0.04, depth_top=0.06, n=21)
    print(f"3-D B-scan: {len(X)} traces in {time.time()-t0:.0f} s "
          f"({(time.time()-t0)/len(X):.1f} s/trace)")
    R = B - B.mean(axis=0, keepdims=True)
    gate = (t > 5e-9) & (t < 12e-9)
    amp = np.abs(R[:, gate]).max(axis=1)
    print("residual amplitude along scan:"); print(np.round(amp, 2))
    print("peak at x =", X[int(np.argmax(amp))], " (target at 0.30)")
