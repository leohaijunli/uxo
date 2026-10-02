# HANDOFF — GPR + magnetic simulation dataset for the OSU seeded UXO set

**Last updated:** 2026-10-01 (v2)  ·  **Working dir:** `/home/dev/Downloads/uxo_detection`
**Sandbox:** workspace-write. Writable = this dir + `/tmp`. Network restricted —
to install/fetch, re-run with `sandbox_permissions="require_escalated"` + `justification`.

---

## 0. THE GOAL (user's own words)

> *"eventually we should have a dataset consisting gpr data and magnetic data as
> well for the simulable objects"*

So the end product is a **multi-modal, per-object dataset**:
every gprMax-simulable object gets
**(a) simulated GPR** and **(b) real measured magnetic data**, plus metadata.
Everything below serves that.

---

## 1. Status

| Item | State |
|---|---|
| Magnetometry detection + 3-D dipole inversion + report | ✅ done (`report_output/`) |
| Classification of which objects gprMax can model | ✅ done (`report_output_gpr/gpr_model_plan.csv`) |
| 2-D TM B-scan pipeline | ✅ validated (`gpr_sim/gpr_bs_scan.py`) |
| **3-D B-scan pipeline** | ✅ **validated** (`gpr_sim/gpr_bs_scan_3d.py`), 4.1 s/trace |
| **Magnetic patches for all 137 simulable objects** | ✅ **done** (`dataset/mag/`, 137 `.npz` + index) |
| **GPR stage of the dataset** | ⬜ **not run yet** — builder ready (`dataset/build_dataset.py --stage gpr3d`) |
| Dataset manifest / packaging / report | ⬜ not started |
| MATLAB/Octave port of the magnetometry report | ⬜ not started (no interpreter; `sudo pacman -S octave` fails — no password) |

---

## 2. Environment — how to run

```bash
cd /home/dev/Downloads/uxo_detection

# analysis env (pandas/numpy/scipy/matplotlib, Python 3.14)
.venv/bin/python  dataset/build_dataset.py --stage mag

# gprMax env (gprMax 4.0.0, Python 3.12)
MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python gpr_sim/gpr_bs_scan_3d.py
```

**Always set `MPLCONFIGDIR=/tmp/mpl`** or matplotlib warns on every import
(`~/.config/matplotlib` is read-only).

- **GPU is NOT usable for gprMax.** RTX 2050 (4 GB), driver 610.57.04, CUDA UMD
  13.3 are installed & healthy, but there is **no CUDA toolkit (`nvcc`)** and
  **no OpenCL ICD** (`/etc/OpenCL/vendors` missing) → `gpu=True`/`opencl=True`
  fail. `sudo` is impossible here (`no_new_privs`). To enable GPU the **user**
  must run `sudo pacman -S cuda opencl-nvidia`. Until then: **CPU/OpenMP only**
  — which is fine, see the timings in §6b.
- `nvidia-smi` needs escalation (`prefix_rule ["nvidia-smi"]` is approved).

---

## 3. Data files

| File | What |
|---|---|
| `EO_properties_and_Locations_field2.csv` | Ground truth, 150 objects (86 ferrous). Cols: `ID, Item, Longitude, Latitude, Class, Depth (cm), Weight (g), Length (cm), Diameter (cm), Ferrous (1) or NF (0), …` |
| `2024-05-09-18-03-43-sensys.csv` | UAV magnetometer, 283 145 rows (`TMI, Bx, By, Bz, Altitude AGL, …`), ~0.5 m line spacing. |
| `2024-05-09-18-03-43-position.csv` | RTK positions, 44 239 rows. |
| `remotesensing-18-02182.pdf` | Baur et al. 2026, *Remote Sensing* 18:2182. |

---

## 4. File inventory

| Path | Contents |
|---|---|
| `invert3d.py` | Magnetometry detection + 3-D dipole inversion. **Importable** (has `if __name__` guard): `lonlat_to_xy`, `load_ground_truth`, `load_magnetometer`, `grid_anomaly`. |
| `make_report.py` | 12-figure magnetometry report generator. |
| `report_output/` | `UXO_detection_report_final.md`, `fig01..fig12*.png`, `uxo_dipoles_v2.csv`, `matched_dipoles_v2.csv`, `uxo_dipole_inversion_v2.mat`, `report_tables.csv`. |
| `gpr_sim/gpr_model_plan.py` | Classifies all 150 CSV objects → gprMax routes. |
| `report_output_gpr/gpr_model_plan.csv` | The classification result (§5). |
| `gpr_sim/gpr_bs_scan.py` | Validated **2-D TM** B-scan generator (fast proxy). |
| `gpr_sim/gpr_bs_scan_3d.py` | Validated **3-D** B-scan generator (faithful). |
| `dataset/build_dataset.py` | **Dataset builder** — stages `mag` / `gpr2d` / `gpr3d`. |
| `dataset/mag/` | **137 magnetic patches** (`<ID>_mag.npz`) + `mag_index.csv`. |
| `dataset/gpr2d/`, `dataset/gpr3d/` | Output dirs, **created but empty** — GPR not yet generated. |

---

## 5. Which CSV objects can be simulated with gprMax

**137 of 150 are simulable; 13 are not.** Full detail in
`report_output_gpr/gpr_model_plan.csv`.

### 5a. `route == "landmine_h5"` — 24 objects use the ready-made
gprMax **LandmineModels** geometry (realistic internal structure):

| Item | gprMax file | n | IDs |
|---|---|---|---|
| `PFM-1` | `PMA_2x2x2.h5` (nearest-shape proxy) | 18 | D1–D18 |
| `PMN` | `PMN_2x2x2.h5` | 2 | D22, D23 |
| `TS-50` | `TS50_2x2x2.h5` | 2 | D24, D25 |
| `Aluminum Can` | `can_2x2x2.h5` | 2 | F7, F9 |

> PFM-1 ≠ PMA physically — PMA is the nearest available plastic scatterable AP
> mine shape. **Flag this assumption** in any writeup.

### 5b. `route == "parametric"` — 113 built from `Length`/`Diameter`:

| material | shape | n | IDs |
|---|---|---|---|
| `pec` (metal) | cylinder | 88 | A1 A3 A5 A7 A9 A11 A13 A15 A17 A21 A25, B1–B25, C1–C20, E1–E18 E20–E25, F5 F11 F13 F15 F17 F19 F21 F23 |
| `pec` (metal) | sphere | 7 | C21–C25, F1, F3 |
| `plastic` | cylinder | 11 | A2 A4 A6 A8 A10 A12 A19 A23, D19 D20 D21 |
| `air` void | sphere | 6 | F6 F8 F10 F12 F14 F16 |
| `explosive` (TNT) | cylinder | 1 | E19 |

### 5c. NOT simulable — 13 objects
`A14 A16 A18 A20 A22 A24 F18 F20 F22 F24 F25` = **blank CSV rows** (no
Item/Class/Length/Diameter). `F2 F4` = Control Holes with depth but **no
diameter**. Do **not** invent dimensions for these.

### 5d. Physics caveat
Metal (95, `pec`) and air voids (6) → strong returns. **Plastic (33) → low
contrast (εr≈2.9 vs soil ≈6) → weak returns** — physically correct, not a bug.
Plastic-bodied scatterables are genuinely hard for GPR.

### 5e. Material values used
```
pec        er=1.0  se=1e7   (metal -> strongest return; use the BUILT-IN "pec")
plastic    er=2.9  se=0.003
explosive  er=3.0  se=0.002  (TNT)
air        er=1.0  se=0      (void)
soil       er=6.0  sm=0.001
```
`target_geometry()` in `dataset/build_dataset.py` converts a plan row to
`(radius_m, length_m, depth_top_m)`.

---

## 6. gprMax 4.0.0 — VALIDATED API recipe + gotchas

**Do not "correct" any of these — they are all verified.**

1. `gprMax.DomainMode(mode="TM")` for 2-D — **not** `"2D TM"`;
   `mode="3D"` for 3-D (the default).
2. **2-D convention:** `Domain(p1=(Lx, Ly, np.inf))` — the **infinite axis is z
   (last)**. Layout: **x = scan direction, y = depth, z = target long axis**.
3. **2-D only: do NOT add `PMLThickness`.** The infinite axis is 1 cell, so
   `PMLThickness(10)` raises `has too many cells for the domain size`. Omit it
   (default PML is applied correctly). In 3-D the default is fine too.
4. Geometry kwarg is **`r`**, not `radius`:
   `Cylinder(p1=..., p2=..., r=..., material_id=...)`,
   `Sphere(p1=..., r=..., material_id=...)`.
5. Use the **built-in material id `"pec"`** for conductors. Declaring
   `Material(er=1, se=1e7, …)` triggers
   `Insufficient spatial resolution in material 'target'`.
6. `Rx(p1=(x,y,z), id="rx", outputs=["Ez"])` — one call, all args together.
7. `gprMax.run(scenes=..., n=..., outputfile=..., hide_progress_bars=True,
   cpu_precision="single", log_level=40)`
   - `cpu_precision` ∈ {`"single"`,`"double"`}; `"mixed"` is rejected.
   - **`i=0` is invalid** (`i must be greater than zero`). Omit `i` entirely.
   - **Multi-trace B-scans:** pass a **list of N `Scene` objects** and
     `n=len(scenes)`. Output files are **suffixed**: `outstem1.h5` …
     `outstemN.h5`. (One scene + `n=N` → `IndexError: list index out of range`.)
   - `SrcSteps`/`RxSteps` exist but only work with input-file parsing, **not**
     with the API scene list — build N scenes instead.
8. **To make a background/empty model,** remove geometry from the scene with
   `sc.geometry_objects[:] = [o for o in sc.geometry_objects if o.hash != "#sphere"]`.
   **`Scene` has no `.objects` attribute** (use `scene.geometry_objects`).
9. Reading output:
   ```python
   with h5py.File("outstem3.h5") as f:
       rx  = list(f["rxs"].keys())[0]          # "rx1"
       ez  = f["rxs"][rx]["Ez"][:]             # (nsamples,)
       pos = f["rxs"][rx].attrs["Position"]    # [x, y, z]
   ```

### 6a. Working 2-D scene (`gpr_sim/gpr_bs_scan.py`)
`SOIL_TOP=0.12 m`, `DL=0.004 m`, Ricker 1.5 GHz, `TW=12 ns`, soil
`er=6, sm=0.001`, Tx 2 cm above surface, Rx offset +0.06 m.
Runtime ≈ **0.15 s/trace**; 31-trace scan ≈ 7 s.

### 6b. Working 3-D scene (`gpr_sim/gpr_bs_scan_3d.py`) — recommended
`SOIL_TOP=0.10 m`, `DL=0.005 m` (→ −2.8 % phase-velocity error, acceptable),
Ricker 1.5 GHz, `TW=12 ns`, domain `0.55×0.25×0.35 m` (~385 k cells).
**Runtime ≈ 4.1 s/trace; a 21-trace 3-D B-scan ≈ 86 s.** Memory is modest —
the earlier fear that 3-D would not fit was **wrong**; 3-D on CPU is affordable.

### 6c. **CRITICAL signal-processing finding**
The **direct Tx–Rx coupling dominates** (~111 V/m in 3-D, ~1855 V/m in 2-D)
while the **target echo is tiny**:

| method | scattered/direct ratio | verdict |
|---|---|---|
| 3-D, **trace-mean** background removal | ~0.04 V/m (noise-level, peaks at scan edge) | ❌ **do not use** |
| 3-D, **explicit no-target background model** (`target − background`) | **1.33 V/m / 111 V/m = 1.2 %** | ✅ **clean, peaks at target** |

➡ **Always subtract an explicit background model run**, not the trace mean.
Then: time-zero correction, AGC/exponential gain, optionally migration.

---

## 7. The dataset builder (`dataset/build_dataset.py`)

```
python dataset/build_dataset.py --stage mag                      # done: 137 patches
python dataset/build_dataset.py --stage gpr3d --ids A3 B1 F6     # pilot
python dataset/build_dataset.py --stage gpr2d                    # fast bulk
```
`--ids` defaults to all 137 simulable objects; `--ntraces` (default 21),
`--dx` (default 0.01 m).

### Layout
```
dataset/
  build_dataset.py
  mag/<ID>_mag.npz         gx, gy (13), anomaly, TMI, Bx, By, Bz  (13x13)
  mag/mag_index.csv        ID,item,cls,x,y,depth_cm,ferrous,mag_anom_peak,mag_anom_rms
  gpr2d/<ID>_bscan.npz     x (n), t (nsamples), Ez (n x nsamples), metadata
  gpr3d/<ID>_bscan.npz     same
  gpr2d|3d/gpr_index.csv
```
Magnetic patches: 0.25 m cells, ±1.5 m window (13×13), centred on the object;
`Bx/By/Bz` are stored as anomalies about their median; the TMI channel is the
rolling-median-baseline anomaly. Verified: **86/137 patches are ferrous**,
matching the ground truth.

### Known open issues in the builder (fix these next)
1. `build_gpr()` currently runs **only the target model** — it must ALSO run a
   **no-target background model** and store `Ez_target − Ez_background`
   (see §6c). Right now the stored `Ez` would be direct-wave-dominated.
2. The 3-D builder uses a **sphere for every object** (`run_bscan_3d(...)` is
   called without `length=`); pass `length` from `target_geometry()` to get a
   **finite cylinder** for elongated UXO. `target_geometry()` already returns it.
3. Per-object geometry changes the domain/depth, so the background model is not
   shareable unless the domain is **canonicalised** (same Lx/Ly/Lz + same source
   line for every object, varying only target radius/depth). Canonicalise it and
   the background run is computed **once**, halving the cost.
4. No de-duplication: many objects are the same item at different depths — fine,
   but consider a `canonical_id` column for grouped splits.

### Full-run cost (measured, 4.1 s/trace)
- 21 traces, target+background = 42 runs ≈ **172 s/object**
- 137 objects ≈ **6.5 h** single-threaded, 4 OpenMP threads.
- With canonicalised background (1 shared run) → **≈3.3 h**.
- `--ntraces 11 --dx 0.02` → **≈1.7 h**.
Run it as a background job; do **not** hold the session open for it.

---

## 8. Immediate next steps (prioritized)

1. **Fix `build_gpr()`** per §7 issues 1–3 (background model + finite cylinders).
2. **Pilot**: `--stage gpr3d --ids A3 B1 C21 F6 E19 D22 F7 A1` (one per class) —
   proves the pipeline end to end in ~25 min.
3. **Figures + report**: B-scans (raw vs background-subtracted vs gained),
   A-scans over/off target, magnetic patch beside the GPR section, and a
   `GPR_simulation_report.md` stating all assumptions (soil εr/σ, Ricker
   1.5 GHz, 5 mm mesh, 2-D proxy vs 3-D, PFM-1→PMA proxy).
4. **Full run** of whichever stage the user wants, as a background job.
5. Package the dataset (`manifest.csv` linking mag + gpr + metadata;
   consider a `.mat`/HDF5 mirror for MATLAB users).
6. Confirm whether the **MATLAB/Octave port** is still wanted.

---

## 9. Hard constraints

- **Do NOT `git commit`** or create branches. Edit files with `apply_patch`.
- Prefer `rg` over `grep`.
- No `sudo` (impossible here). If an install is truly needed, ask the user.
- Network is sandboxed; escalate explicitly when required.
- Don't invent dimensions for the 13 non-simulable rows.
- **LandmineModels licence: CC BY-SA 4.0** — cite **Giannakis, Giannopoulos &
  Warren (2016), IEEE JSTARS 9(1):37–51**.
- Detection-eval caveat from the magnetometry work: seeded targets are ~1.02 m
  apart, so recall at ≥1 m tolerance is saturated — use **R = 0.5 m + a fixed
  budget + a random baseline** for fair comparison.

---

## 10. Smoke tests

```bash
# 2-D B-scan (7 s)
MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python gpr_sim/gpr_bs_scan.py
#   -> 31 residual amplitudes, peak near x=0.50 (the target)

# 3-D B-scan (86 s)
MPLCONFIGDIR=/tmp/mpl .venv-gpr/bin/python gpr_sim/gpr_bs_scan_3d.py
#   -> 21 traces, 4.1 s/trace, residual amplitude profile

# magnetic dataset (already built)
.venv/bin/python -c "import pandas as pd;print(pd.read_csv('dataset/mag/mag_index.csv').shape)"
#   -> (137, 9)
```
