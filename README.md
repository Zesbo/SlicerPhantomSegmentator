# Phantom Segmentator

A [3D Slicer](https://www.slicer.org) scripted module for semi-automatic segmentation of CT calibration phantoms and HU→mgCaHA calibration. Developed at the [ZESBO](https://www.uni-leipzig.de/zesbo), University of Leipzig.

## Overview

This module supports a quantitative CT (QCT) calibration workflow using physical bone-equivalent phantoms. It segments cylindrical rod inserts in CT volumes, computes a linear HU→mgCaHA calibration curve, and generates vBMD volumes. Four phantom geometries are supported out of the box.

### Supported Phantoms

| Phantom | Type | Rods | Registration strategy |
|---------|------|------|-----------------------|
| BDC-6H200 (Reference) | Commercial reference standard | 6 | Region-growing + axis landmarks |
| uCT_4r_V2 | 3D-printed PolyJet, square arrangement | 4 | User-drawn markup lines |
| 3Dprinted_6H100_V1 | 3D-printed PolyJet, linear arrangement | 6 | Region-growing + axis landmarks |
| uCT_6H_V1 | 3D-printed PolyJet, curved arrangement | 6 | Region-growing + axis landmarks |

## Requirements

- **3D Slicer** ≥ 5.0 (tested on 5.6)
- **SegmentStatistics** extension (bundled with Slicer)
- **numpy** (bundled with Slicer)
- **scipy** ≥ 1.7 (optional — required for the *Remove Voids* feature only)

## Installation

### Option A — Manual (development)

1. Clone or download this repository.
2. In 3D Slicer: **Edit → Application Settings → Modules → Additional module paths** → add the folder containing `Phantom_Segmentator.py`.
3. Restart Slicer. The module appears under **Modules → Zesbo → Phantom Segmentator**.

### Option B — Extension Manager (planned)

Submission to the official Slicer Extension Catalog is planned for a future release.

## Usage

1. **Load** a CT volume containing the phantom into Slicer.
2. Open **Phantom Segmentator** (Modules → Zesbo).
3. **Select phantom type** from the dropdown (loads the corresponding geometry JSON).
4. **Select CT volume** from the volume dropdown — the volume is immediately shown in all 2D slice views.
5. Click **"Place Seeds"** (or *"Place Markups (2 lines)"* for uCT_4r_V2):
   - Place seed points inside the registration rods as instructed.
6. Click **"Segment Phantom"** — the module performs:
   - Region-growing segmentation of registration rods
   - PCA-based axis and endpoint estimation
   - Rigid landmark registration (phantom coordinates → CT coordinates)
   - Cylinder fill of all rods from the registered geometry
7. Visually verify the segmentation. If the phantom orientation appears flipped, click **"Flip"**.
8. *(Optional)* Click **"Remove Voids"** to remove air void voxels (HU < −250) from segments.
9. Click **"Regression / Statistics"** to run Segment Statistics and — for phantoms with reference values (BDC-6H200, uCT_4r_V2, 3Dprinted_6H100_V1) — a weighted linear regression (mgCaHA = m·HU + b). For the cross-calibrated 3D-printed phantoms, run *Remove Voids* first. STANDARD-kernel reference values are used by default; BONEPLUS values are used automatically when the volume name contains "bone".
10. Click **"Create vBMD Volume"** to produce a new volume with calibrated vBMD values (requires prior regression).

## Phantom Geometry Files

Geometry is defined in JSON files under `Resources/Geometry/`. Each file specifies rod axis endpoints, radius, HU thresholds for region-growing, and material composition.

| File | Phantom |
|------|---------|
| `PHANTOM_GEOMETRY_BDC-6H200.json` | BDC-6H200 reference phantom |
| `PHANTOM_GEOMETRY_uCT_4r_V2.json` | uCT_4r_V2 |
| `PHANTOM_GEOMETRY_3Dprinted_6H100_V1.json` | 3Dprinted_6H100_V1 |
| `PHANTOM_GEOMETRY_uCT_6H_V1.json` | uCT_6H_V1 |

Optional keys:

- `phantom_theory` — reference values per rod (`caHA_mgcm3`, `sigma_mgcm3`); enables the regression. For uCT_4r_V2 and 3Dprinted_6H100_V1 these are the STANDARD-kernel cross-calibration values against BDC-6H200.
- `phantom_theory_bone` — BONEPLUS-kernel reference values, selected when the volume name contains "bone".
- `remove_voids` — shows the *Remove Voids* button (3D-printed phantoms).

To add a custom phantom, create a new JSON following the same structure and add it to `PHANTOM_CONFIG` in `Phantom_Segmentator.py`.

## Citation

If you use this software in your research, please cite the accompanying paper:

> Heilmann R, *et al.* (2026). *3D-printed bone calibration phantoms as transferable QCT standards.* Physica Medica. [DOI pending]

BibTeX:
```bibtex
@article{heilmann2026phantom,
  author  = {Heilmann, Robin and others},
  title   = {3D-printed bone calibration phantoms as transferable QCT standards},
  journal = {Physica Medica},
  year    = {2026},
  doi     = {pending}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Developed at the Centre for Spine and Bone Oncology (ZESBO), University of Leipzig, Germany.
Funded by [funding source].
