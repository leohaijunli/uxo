"""
Self-contained GPR B-scan builder for the gprMax env (numpy + csv only, no
pandas/scipy, which the gprMax 3.12 venv lacks).

Mirrors build_dataset.build_gpr() but avoids importing pandas/scipy/invert3d.
Runs a SINGLE shared canonical no-target background model and stores the
background-subtracted field Ez = Ez_target - Ez_background (HANDOFF section 6c).

    MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python dataset/build_gpr_only.py \
        --stage gpr3d --ids A3 B1 C21   # pilot
        --stage gpr3d                   # all simulable objects
        --stage gpr2d --ntraces 31      # fast proxy
"""
import os, sys, csv, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gpr_sim"))

PLAN_CSV = os.path.join(ROOT, "report_output_gpr", "gpr_model_plan.csv")


def load_plan():
    with open(PLAN_CSV) as f:
        rows = list(csv.DictReader(f))
    return {r["ID"]: r for r in rows}


def simulable_ids(plan):
    return [i for i, r in plan.items() if r["feasible"] in ("yes", "limited")]


def target_geometry(row):
    D, L = row.get("Diameter_cm") or "", row.get("Length_cm") or ""
    def num(v):
        try:
            return float(v)
        except ValueError:
            return float("nan")
    D, L = num(D), num(L)
    if np.isfinite(D):
        r = D / 200.0
    elif np.isfinite(L):
        r = L / 200.0
    else:
        r = 0.10
    depth = num(row.get("depth_cm") or "") / 100.0 if num(row.get("depth_cm") or "") else 0.05
    if str(row.get("model")) == "sphere":
        length = None
    elif np.isfinite(L):
        length = min(L / 100.0, 0.10)
    else:
        length = None
    return float(r), length, float(depth)


def build(ids, mode, ntraces, dx):
    out = os.path.join(HERE, f"gpr{mode}")
    os.makedirs(out, exist_ok=True)
    plan = load_plan()

    if mode == "3d":
        from gpr_bs_scan_3d import run_bscan_3d
        _, Bg, _t = run_bscan_3d("/tmp/uxo/bg_3d", xc=0.30, r=0.05, depth_top=0.05,
                                 n=ntraces, dx=dx, add_target=False)
        def run_target(st, r, depth, length):
            return run_bscan_3d(st, xc=0.30, r=r, depth_top=min(depth, 0.15),
                                length=length, n=ntraces, dx=dx)
    else:
        from gpr_bs_scan import run_bscan
        _, Bg, _t = run_bscan("/tmp/uxo/bg_2d", xc=0.50, radius=0.05, depth_top=0.05,
                              n=ntraces, dx=dx, add_target=False)
        def run_target(st, r, depth, length):
            return run_bscan(st, xc=0.50, radius=r, depth_top=min(depth, 0.15),
                             n=ntraces, dx=dx)

    rows = []
    for i, idv in enumerate(ids, 1):
        row = plan[idv]
        r, length, depth = target_geometry(row)
        try:
            X, B, t = run_target(os.path.join("/tmp/uxo", f"ds_{idv}"),
                                 r, depth, length)
            Ez = B - Bg
        except Exception as e:
            print(f"  !! {idv}: {e}")
            rows.append(dict(ID=idv, ok=False))
            continue
        np.savez_compressed(os.path.join(out, f"{idv}_bscan.npz"),
                            x=X, t=t, Ez=Ez, ID=idv, item=str(row.get("Item")),
                            material=row.get("material"), radius_m=r,
                            length_m=length, depth_top_m=depth, mode=mode,
                            bg_subtracted=True)
        rows.append(dict(ID=idv, ok=True, item=str(row.get("Item")),
                         material=row.get("material"), radius_m=r,
                         length_m=length, depth_top_m=depth, ntraces=len(X)))
        Ls = "None" if length is None else round(length, 2)
        print(f"  [{i}/{len(ids)}] {idv} {str(row.get('Item'))[:22]:22s} "
              f"r={r:.3f} d={depth:.2f} L={Ls} -> {out}/{idv}_bscan.npz")

    keys = ["ID", "ok", "item", "material", "radius_m", "length_m",
            "depth_top_m", "ntraces"]
    with open(os.path.join(out, "gpr_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rw in rows:
            w.writerow({k: rw.get(k, "") for k in keys})
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"gpr{mode}: {ok}/{len(rows)} objects written -> {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="gpr3d", choices=["gpr2d", "gpr3d"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--ntraces", type=int, default=11)
    ap.add_argument("--dx", type=float, default=0.02)   # from real speed/PRF
    A = ap.parse_args()
    plan = load_plan()
    ids = A.ids or simulable_ids(plan)
    build(ids, mode=A.stage[-2:], ntraces=A.ntraces, dx=A.dx)