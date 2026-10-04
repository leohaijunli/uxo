"""
Build the 3-D grid GPR dataset for the 8-object pilot (one per class).

Each object gets a true 3-D survey: an nx x ny antenna grid (dx=dy=0.02 m from
the real drone speed/PRF) at 0.05 m ground-coupled height. The result is a
background-subtracted cube Ez[nx, ny, t] saved to dataset/gpr3d_grid/.

Objects routed to gprMax LandmineModels (PMN, can, TS50, PMA) use the built-in
HDF5 target models via GeometryObjectsRead; all others use parametric
cylinder/sphere geometry.

Run (background):  MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python \
                       dataset/build_gpr3d_grid.py
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gpr_sim"))
import gpr_bs_scan_3d as g3   # noqa

LM = os.path.join(ROOT, ".venv-gpr/lib/python3.12/site-packages/gprMax/"
                  "toolboxes/LandmineModels")
PLAN_CSV = os.path.join(ROOT, "report_output_gpr", "gpr_model_plan.csv")
OUT = os.path.join(HERE, "gpr3d_grid")

PILOT = ["A3", "B1", "C21", "F6", "E19", "D22", "F7", "A1"]
NX = NY = 11
DX = DY = 0.02
XC, YC = 0.30, 0.125

# material -> (er, se).  'pec' uses the built-in conductor id.
MAT = {"pec": ("pec", 1.0, 0.0),
       "plastic": ("plastic", 2.9, 0.003),
       "explosive": ("explosive", 3.0, 0.002),
       "air": ("air", 1.0, 0.0)}


def load_plan():
    with open(PLAN_CSV) as f:
        return {r["ID"]: r for r in csv.DictReader(f)}


def geom(row):
    D, L = row.get("Diameter_cm") or "", row.get("Length_cm") or ""
    def num(v):
        try:
            return float(v)
        except ValueError:
            return float("nan")
    D, L = num(D), num(L)
    r = D / 200.0 if np.isfinite(D) else (L / 200.0 if np.isfinite(L) else 0.05)
    length = None if str(row.get("model")) == "sphere" else \
        (min(L / 100.0, 0.10) if np.isfinite(L) else None)
    depth = num(row.get("depth_cm") or "") / 100.0 if num(row.get("depth_cm") or "") else 0.05
    return float(r), length, float(depth)


def landmine_file(model):
    name = {"PMA": "PMA_2x2x2", "PMN": "PMN_2x2x2", "TS50": "TS50_2x2x2",
            "can": "can_2x2x2"}[model]
    return os.path.join(LM, f"{name}.h5")


def build(ids=None, nx=NX, ny=NY):
    os.makedirs(OUT, exist_ok=True)
    plan = load_plan()
    ids = ids or PILOT
    rows = []
    # cache one parametric background and one landmine background (per model)
    bgcache = {}
    for i, idv in enumerate(ids, 1):
        row = plan[idv]
        route, model = row.get("route"), str(row.get("model"))
        r, length, depth = geom(row)
        print(f"[{i}/{len(ids)}] {idv} {str(row.get('Item'))[:22]} "
              f"route={route} r={r:.3f} depth={depth:.2f}", flush=True)
        try:
            if route == "landmine_h5":
                gf = landmine_file(model)
                def mk(xs, ys, add_target=True):
                    return g3.make_scene_landmine(
                        xs, ys, XC, YC, depth_top=depth, geofile=gf,
                        add_target=add_target)
                bkey = f"lm_{model}"
            else:
                mat = row.get("material")
                mid, er, se = MAT.get(mat, (mat, 1.0, 0.0))
                def mk(xs, ys, add_target=True):
                    return g3.make_scene(xs, ys, XC, YC, r, length=length,
                                         depth_top=depth, material_id=mid,
                                         er_obj=er, se_obj=se,
                                         add_target=add_target)
                bkey = "parametric"
            if bkey not in bgcache:
                bgf = f"/tmp/uxo/grid/bg_{bkey}.npz"
                if os.path.exists(bgf):
                    z = np.load(bgf); bgcache[bkey] = z["Bg"]
                    print(f"   bg[{bkey}] loaded from cache", flush=True)
                else:
                    X, Y, Bg, t = g3.run_grid_3d(f"/tmp/uxo/grid/bg_{bkey}",
                                                 nx, ny, DX, DY, XC, YC,
                                                 lambda a, b: mk(a, b, add_target=False))
                    np.savez_compressed(bgf, Bg=Bg)
                    bgcache[bkey] = Bg
                    print(f"   bg[{bkey}] done + cached", flush=True)
            X, Y, Bt, t = g3.run_grid_3d(f"/tmp/uxo/grid/{idv}",
                                        nx, ny, DX, DY, XC, YC, mk)
            Ez = Bt - bgcache[bkey]
            np.savez_compressed(os.path.join(OUT, f"{idv}_cube.npz"),
                                Ez=Ez, X=X, Y=Y, t=t, ID=idv,
                                item=str(row.get("Item")), material=row.get("material"),
                                radius_m=r, length_m=length, depth_top_m=depth,
                                xc=XC, yc=YC, mode="3d_grid", bg_subtracted=True)
            rows.append(dict(ID=idv, item=str(row.get("Item")),
                             material=row.get("material"), route=route,
                             model=model, radius_m=r, depth_top_m=depth,
                             nx=nx, ny=ny, dx=DX, dy=DY, ok=True))
            print(f"   saved {idv}_cube.npz peak_sub={np.abs(Ez).max():.3f}",
                  flush=True)
        except Exception as e:
            print(f"   !! {idv}: {e}", flush=True)
            rows.append(dict(ID=idv, ok=False, route=route, model=model))
    with open(os.path.join(OUT, "gpr_index.csv"), "w", newline="") as f:
        keys = ["ID", "item", "material", "route", "model", "radius_m",
                "depth_top_m", "nx", "ny", "dx", "dy", "ok"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for rw in rows:
            w.writerow({k: rw.get(k, "") for k in keys})
    print(f"gpr3d_grid: {sum(1 for r in rows if r.get('ok'))}/{len(rows)} -> {OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--ny", type=int, default=None)
    A = ap.parse_args()
    build(ids=A.ids, nx=A.nx, ny=A.ny or A.nx)