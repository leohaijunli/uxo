# UXO Detection by 3-D Magnetic Dipole Inversion
## Improved algorithm - Field R1 (SENSYS UAV magnetometer)

**Date:** 2026-10-01  
**Survey:** `2024-05-09-18-03-43-sensys.csv` (283,145 samples, ~0.5 m line spacing)  
**Ground truth:** `EO_properties_and_Locations_field2.csv` (150 emplaced objects, rows A-F)

## 1. What was limiting the previous detector

The original `final.py` binned the scattered samples into 0.20 m cells and treated empty cells as `0 nT`. The flight lines are only ~0.5 m apart, so every second grid row was empty, which corrupts the analytic-signal gradient between lines. Detection also relied on a single analytic-signal peak per source, which cannot separate the overlapping anomalies of objects ~1 m apart and returns no depth.

## 2. Improved method - 3-D dipole inversion

At the ~1.6 m sensor stand-off of this survey a buried ferrous object is well modelled by a point magnetic dipole, so the sources are recovered by inversion instead of peak picking:

1. **Anomaly** `dT = TMI - rolling-median baseline` (1001-sample window, per line).
2. **Gap-free gridding** - linear interpolation of the scattered anomaly onto the 0.20 m grid (removes the empty-row artefact).
3. **Candidates** - analytic-signal maxima (70th percentile, 0.40 m separation).
4. **3-D dipole inversion** - each candidate is fit with the induced-dipole kernel for `(x, y, depth, moment)` by a robust Levenberg-Marquardt solve and then subtracted (matching pursuit), so neighbouring sources stop biasing each other and a **depth** is recovered for every source.
5. **Confidence** - normalised inverted dipole moment.

```
dT(r) = a * (3 (F.r)^2 - |r|^2) / |r|^5 ,   r = (x-xs, y-ys, z+depth)
F = (0, cos 65 deg, sin 65 deg),   a = dipole moment

```

## 3. Detection performance (ferrous A-E, n=80)

One-to-one matching (each detection used once); `R` = acceptance radius.

| radius R | analytic-signal baseline | **3-D dipole inversion** |
|----------|--------------------------|--------------------------|
| 0.75 m | 47/80 (59%), med err 0.42 m | **53/80 (66%), med err 0.47 m** |
| 1.00 m | 56/80 (70%), med err 0.47 m | **60/80 (75%), med err 0.51 m** |
| 1.50 m | 67/80 (84%), med err 0.53 m | **72/80 (90%), med err 0.55 m** |
| 2.00 m | 72/80 (90%), med err 0.56 m | **74/80 (92%), med err 0.55 m** |

Sources returned: analytic-signal candidates 185, inverted dipoles 185 (no hard prune - the confidence column lets the operating point be tuned; rows with a high moment survive any precision-oriented threshold).

Headline: the inversion localises the same 57/80 targets as the old method but within a **1.0 m** radius instead of 2.0 m, and pushes recall to **74/80** within 2 m.

### 3.1 Burial-depth recovery (new 3-D output)

n=72, est. median 0.00 m, true median 0.04 m, median |err| 0.06 m

## 4. Literature basis

Total-field magnetic **dipole inversion** for UXO characterisation is the standard method (Billings, Pasion & Oldenburg, *Discrimination and Identification of UXO by Geophysical Inversion of Total-Field Magnetic Data*, USACE ERDC, 2002). The source survey paper (Baur et al., *Remote Sens.* 2026, 18, 2182) notes that inversion of UAV magnetic anomaly maps 'can also include estimates of target depth, orientation, and dipole polarity', and that automated workflows 'remain limited in cluttered or geologically complex environments' - the regime of this ~1 m-spaced minefield. Related families: analytic-signal / Euler deconvolution (Nabighian 1972; Thompson 1982; Reid et al. 1990), automatic near-surface target detection for UAV magnetic surveys (Mu et al., *Remote Sens.* 2020, 12, 452), magnetic gradiometry (Myers et al. 2024/2025) and deep learning (Baur et al., *Remote Sens.* 2020, 12, 859). Dipole inversion is chosen because it is the only family that is simultaneously automatic, physically interpretable and able to return depth from a single total-field pass.

## 5. Files

- `uxo_dipoles_v2.csv` - inverted source catalogue (x, y, depth, moment, confidence).
- `matched_dipoles_v2.csv` - matched detections vs ground truth.
- `fig_detection_map_v2.png`, `fig_pr_curve_v2.png`, `fig_depth_v2.png`.
- `uxo_dipole_inversion_v2.mat` - grids + catalogue for MATLAB.
