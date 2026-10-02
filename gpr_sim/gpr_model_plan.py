"""
Decide which objects in EO_properties_and_Locations_field2.csv can be turned into
gprMax GPR models, and how.

Two routes:
  * "landmine_h5"  - use the ready-made realistic geometry from the gprMax
                     LandmineModels toolbox (PMN, PMA, TS50, can).
  * "parametric"   - build the object from the CSV Length/Diameter as a
                     cylinder/sphere with a material chosen from its metal /
                     plastic / void class.

Outputs report_output_gpr/gpr_model_plan.csv and prints a summary.
"""
import os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report_output_gpr")
os.makedirs(OUT, exist_ok=True)

# ---- materials (gprMax: er, se, mr, sm) ---------------------------------- #
MATERIALS = {
    "pec":      dict(er=1.0, se=1e7, mr=1.0, sm=0.0, kind="metal"),     # perfect conductor
    "plastic":  dict(er=2.9, se=0.003, mr=1.0, sm=0.0, kind="plastic"),
    "explosive":dict(er=3.0, se=0.002, mr=1.0, sm=0.0, kind="explosive"),
    "wood":     dict(er=2.5, se=0.002, mr=1.0, sm=0.0, kind="wood"),
    "air":      dict(er=1.0, se=0.0, mr=1.0, sm=0.0, kind="void"),
}

# ---- ready-made gprMax LandmineModels geometries (attributes in mm) ------- #
LANDMINE_H5 = {"PMN": "PMN_2x2x2.h5", "PMA": "PMA_2x2x2.h5",
               "TS50": "TS50_2x2x2.h5", "can": "can_2x2x2.h5"}

# ---- keyword -> (material, landmine model or None) ------------------------ #
# LandmineModels give a *realistic* internal structure for these four targets.
ITEM_RULES = [
    (r"^PMN$",                    "plastic",   "PMN"),
    (r"^TS-?50$",                 "plastic",   "TS50"),
    (r"^PMA",                     "plastic",   "PMA"),
    (r"PFM-?1",                   "plastic",   "PMA"),    # PFM-1 ~ PMA-1 (scatterable plastic AP mine)
    (r"Aluminum Can|Aluminium Can","pec",       "can"),
    (r"PVC|Plastic|3D ?Print",    "plastic",   None),
    (r"VPMA|OZM",                 "plastic",   None),
    (r"TNT",                      "explosive", None),
    (r"Phone|Huawei|HTC",         "pec",       None),     # electronics ~ metallic clutter
    (r"Wood",                     "wood",      None),
]
PLASTIC_CLASSES = {"3D Printed", "Scatterable Anti-personnel", "Anti-personnel Blast"}
VOID_CLASSES = {"Control Hole"}
NONMETAL_ITEMS = ("PVC", "Plastic", "OZM", "VPMA", "TNT", "Wood", "Hole")


def parse_size_from_name(item):
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", item)
    return float(m.group(1)) if m else np.nan


def classify(row):
    item = "" if pd.isna(row["Item"]) else str(row["Item"]).strip()
    cls = "" if pd.isna(row["Class"]) else str(row["Class"]).strip()
    # 1) item-level overrides (incl. the toolbox models)
    for pat, mat, lm in ITEM_RULES:
        if re.search(pat, item, re.I):
            return mat, lm
    # 2) class-level defaults
    if cls in VOID_CLASSES or re.match(r"^Hole", item, re.I):
        return "air", None
    if cls in PLASTIC_CLASSES:
        return "plastic", None
    if any(k.lower() in item.lower() for k in NONMETAL_ITEMS):
        return "plastic", None
    # 3) everything else with a metal-like class -> conductor
    return "pec", None


def main():
    a = pd.read_csv(os.path.join(ROOT, "EO_properties_and_Locations_field2.csv"),
                    encoding="utf-8-sig")
    a = a[a.ID.notna()].copy()
    a["ID"] = a["ID"].astype(str).str.strip()
    for c in ["Depth", "Length (cm)", "Diameter (cm)", "Weight (g)"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")

    rows = []
    for _, r in a.iterrows():
        mat, lm = classify(r)
        L, D, dep = r["Length (cm)"], r["Diameter (cm)"], r["Depth"]
        # holes carry their size in the item name (e.g. "Hole 15cm")
        if pd.isna(D) and mat == "air":
            D = parse_size_from_name(str(r["Item"]))
        # caches / aggregated clutter have no single size: assume a 20 cm box
        if pd.isna(D) and pd.isna(L) and re.search(r"cache", str(r["Item"]), re.I):
            D = 20.0
        # geometry proxy: use diameter if present else length; radius/length in cm
        vals = [v for v in [D, L] if pd.notna(v)]
        size = np.nanmax(vals) if vals else np.nan
        has_dims = pd.notna(L) or pd.notna(D)
        if lm:
            route, model = "landmine_h5", lm
        elif not has_dims:
            route, model = "none", "-"
        elif pd.notna(D) and abs((L if pd.notna(L) else D) - D) < 1e-6:
            route, model = "parametric", "sphere"
        else:
            route, model = "parametric", "cylinder"
        # feasibility
        if not has_dims and not lm:
            feas, why = "no", "no Length/Diameter in CSV"
        elif mat == "air":
            feas, why = "yes", "air void in soil (strong contrast)"
        elif mat == "pec":
            feas, why = "yes", "metal (PEC) - strongest GPR return"
        elif mat == "plastic":
            feas, why = ("limited" if (pd.notna(size) and size < 6) else "yes",
                         "low dielectric contrast (weak return)")
        elif pd.notna(size) and size < 4:
            feas, why = "limited", "smaller than ~lambda/4"
        else:
            feas, why = "yes", ""
        rows.append(dict(ID=r["ID"], Item=r["Item"], Class=r["Class"],
                         material=mat, m_kind=MATERIALS[mat]["kind"],
                         Length_cm=L, Diameter_cm=D, depth_cm=dep,
                         route=route, model=model, feasible=feas, note=why))

    plan = pd.DataFrame(rows)
    plan.to_csv(os.path.join(OUT, "gpr_model_plan.csv"), index=False)

    # ---------------------------- summary ------------------------------- #
    print("total objects:", len(plan))
    for key, grp in plan.groupby("route"):
        sub = "  feasible: " + str(grp.feasible.value_counts().to_dict())
        print(f"route {key:12s}: {len(grp):3d}   {sub}")
    print("\nby material:")
    print(plan.groupby(["material", "feasible"]).size().to_string())
    print("\nitems using the ready-made gprMax LandmineModels:")
    lm = plan[plan.route == "landmine_h5"]
    print(lm.groupby(["Item", "model"]).size().to_string())
    print("\nfeasibility counts:", plan.feasible.value_counts().to_dict())
    print("\nobjects NOT simulable:")
    print(plan[plan.feasible == "no"][["ID", "Item", "Class", "note"]].to_string(index=False))
    return plan


if __name__ == "__main__":
    main()
