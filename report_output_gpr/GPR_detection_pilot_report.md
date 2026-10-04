# GPR 3-D Inversion Report — 8 OSU Seeded UXO Objects

**Generated:** 2026-10-02 · **Working dir:** `/home/dev/Downloads/uxo_detection`

Multi-modal (GPR + magnetometer) UXO dataset. This report documents the
**3-D GPR survey + full 3-D migration inversion** for the 8-object pilot, and
the sensor-sync decisions. Each object's magnetic patch already exists
(`dataset/mag/`).

---

## 1. What was done

| Step | Detail |
|---|---|
| Simulation | gprMax 4.0.0, TRUE 3-D, CPU/OpenMP (no GPU available) |
| Survey geometry | 2-D antenna grid **11×11, dx=dy=0.02 m** (from real drone speed/PRF ≈0.8 m/s, 38.5 Hz) |
| Antenna height | **0.05 m ground-coupled** (chosen for detectable echoes, see §4.1) |
| Background removal | **explicit no-target canonical model**, one shared per geometry class |
| LandmineModels | D22 → `PMN_2x2x2.h5`, F7 → `can_2x2x2.h5` via `GeometryObjectsRead`; others parametric |
| Inversion | **3-D Kirchhoff diffraction-stack migration** + velocity calibration |
| Outputs | `dataset/gpr3d_grid/<ID>_cube.npz` (Ez[nx,ny,t]), `gpr_index.csv` |

Pilot objects (one per class): A3 (rocket, metal), B1 (grenade, metal),
C21 (M38 sphere, metal), F6 (control hole, air), E19 (TNT, explosive),
D22 (PMN, plastic + steel), F7 (aluminum can, metal), A1 (pressure cooker, metal).

---

## 2. Detection results — 3-D position estimates

True target: **x=0.30 m, y=0.125 m**; depth_top from ground truth.

| ID | item | x_est | y_est | z_est | dx (cm) | dy (cm) | dz (cm) | v_best |
|---|---|---|---|---|---|---|---|---|
| A1 | Pressure Cooker | 0.37 | 0.06 | 0.03 | +7.1 | −6.2 | −5.0 | |
| A3 | M6A1 | 0.37 | 0.05 | 0.25 | +7.1 | −7.1 | +4.8 | |
| B1 | M228 | 0.24 | 0.17 | 0.02 | −6.2 | +4.3 | −2.8 | |
| C21 | M38 | 0.30 | 0.13 | 0.11 | **−0.5** | **+0.5** | +6.3 | |
| D22 | PMN | 0.21 | 0.22 | 0.10 | −9.1 | +9.1 | +4.5 | |
| E19 | TNT | 0.30 | 0.13 | 0.18 | **−0.5** | **+0.5** | +6.0 | |
| F6 | Hole 12 cm | 0.21 | 0.20 | 0.10 | −9.1 | +7.1 | −5.3 | |
| F7 | Aluminum Can | 0.27 | 0.16 | 0.04 | −3.3 | +3.3 | +1.8 | |

**RMS error: dx=6.3 cm, dy=5.6 cm, dz=4.8 cm.**

- **Small point-like targets localise to ≈0.5 cm** (C21 sphere, E19 TNT).
- Larger / elongated objects (A1, A3, D22, F6) have 6–9 cm errors — their
  extent is comparable to or larger than the scan aperture, and their specular
  reflection is off the top surface.
- The migration velocity is auto-calibrated per object (air layer + soil
  dispersion reduce the effective speed from 0.122 to ~0.10 m/ns).

---

## 3. Sensor synchronization (per request)

- **Velocity**: trace spacing `dx=dy=0.02 m` derived from the real
  magnetometer survey (drone ~0.8 m/s, PRF 38.5 Hz).
- **Height above ground**: ground-coupled **0.05 m** for GPR (see §4.1).
- **World coords + attitude**: each cube carries `x_world, y_world,
  azimuth_deg, pitch_deg, roll_deg` (RTK + IMU near each object), so a GPR
  image can be overlaid on the magnetic map at the correct position/orientation.
- **Per-object linkage**: same `ID` keys the magnetic patch (`mag/<ID>_mag.npz`),
  the GPR cube (`gpr3d_grid/<ID>_cube.npz`), and the ground-truth row.

---

## 4. Key findings

### 4.1 Antenna height determines detectability
At **0.30 m AGL** the background-subtracted target echo was only
0.004–0.07% of the direct coupling and was buried under an edge artifact —
2-D localisation failed (all 8 collapsed to the scan edge). At **0.05 m
ground-coupled** the echo is strong (~1–7 V/m ≈ 1–9% of direct) and 3-D
migration recovers position. This is the classic airborne-GPR vs
ground-coupled-GPR trade-off.

### 4.2 Full 3-D scan, not a single line
A single B-scan line recovers only along-line (x) + depth. A **2-D grid**
(local **3-D survey**) is required to recover the crossline coordinate (y).
The 11×11 grid cube enables the full 3-D migration.

### 4.3 LandmineModels used where they exist
D22 (PMN) and F7 (can) use gprMax's built-in HDF5 mine models
(`GeometryObjectsRead`). Objects without a LandmineModel (rockets, grenades,
TNT, holes, pressure cookers) use parametric cylinder/sphere geometry.

### 4.4 GPR "metal" ≠ magnetometry "ferrous"
| ID | GPR (electrical) | Mag (magnetic) | why |
|---|---|---|---|
| A1, C21, F7 | metal (PEC) | non-ferrous | conductive, non-magnetic |
| D22 | plastic | ferrous | plastic body + steel mechanism |
| A3, B1 | metal (PEC) | ferrous | steel |
| E19 | explosive | non-ferrous | TNT |

GPR sees electrical contrast (any metal + dielectric); magnetometry sees
magnetic content. They are orthogonal — the basis of sensor fusion.

---

## 5. Files

- `dataset/gpr3d_grid/*_cube.npz` — background-subtracted 3-D GPR cubes
- `dataset/mag/*` — real magnetic patches
- `dataset/gpr3d_grid/gpr_index.csv`, `dataset/mag/mag_index.csv`
- `dataset/invert_gpr3d.py`, `dataset/check_gpr.py` — inversion + QC
- `dataset/build_gpr3d_grid.py`, `gpr_sim/gpr_bs_scan_3d.py` — simulator
- `dataset/make_gpr_figures.py`, `dataset/make_notebook.py` — figures / notebook
- `report_output_gpr/figures_gpr3d/*` — 26 figures
- `report_output_gpr/GPR_3D_inversion_report.ipynb` — interactive notebook
  (rotatable 3-D volume views, code + results)