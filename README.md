# Change Detection — Intrusion Detection System

**Author:** Ugolini Filippo — 0001146279  
**Course:** Image Processing and Computer Vision (IPCV)  
**University:** Università di Bologna

---

## Overview

A classical (non-deep-learning) intrusion detection system for static-camera surveillance footage. The algorithm detects three types of events in real time:

| Label | Color | Event |
|-------|-------|-------|
| `PERSON` | Green | A person enters or moves in the scene |
| `ADDED` | Red | An object is placed in the scene that was not there before |
| `REMOVED` | Cyan | An object previously present has been taken away |

All detection is based on background subtraction, morphological processing, and hand-crafted feature engineering — no neural networks involved.

---

## Algorithm Architecture

The pipeline runs three parallel detection passes on every frame:

```
VIDEO FRAME
    │
    ▼
[PRE-PROCESSING]
  • Grayscale → CLAHE (contrast enhancement)
  • Gaussian blur 5×5
    │
    ▼
[CHANGE DETECTION]
  • Intensity diff:  |bg - frame| > 18
  • Edge diff:       |Sobel(bg) - Sobel(frame)| > median(diff_edge)
  • fg = intensity AND edge  (dual condition, reduces noise)
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
[PASS 1 — PERSON]                     [PASS 3 — REMOVED OBJECTS]
  • Heavy morphology (9×9 close ×5)     • Uses frozen static_bg (never updated)
  • classify_person(): area, AR,         • Detects missing objects even after
    solidity, extent, spatial gate         adaptive bg has absorbed the signal
  → person_mask                          → removed_dets
    │
    ▼
[PASS 2 — ADDED OBJECTS]
  • fg minus person_mask
  • Edge-direction filter (static_bg):
    exclude pixels where static_bg had
    more structure than current frame
    (correctly handles dark ADDED objects)
  • reject_object(): border, solidity
  → obj_dets
    │
    ▼
[TRACKING]
  • IoU-based association (threshold 0.2)
  • Centroid-distance fallback (80 px)
  • Confirmation: 15 frames (ADDED), 20 frames (REMOVED)
  • Confirmed ADDED tracks persist 60 frames when occluded
    │
    ▼
[SELECTIVE BACKGROUND UPDATE]
  • Block update where ADDED objects are present
  • Fast update (α×8) where REMOVED objects are confirmed
    │
    ▼
[OUTPUT]
  CSV + annotated video + per-frame debug images
```

---

## Key Design Decisions

| Problem | Solution |
|---------|----------|
| Adaptive bg absorbs removed-object signal in ~70 frames | Frozen `static_bg` for PASS 3 — never updated |
| Dark ADDED object on light floor filtered as "removal" | Edge-direction filter uses `sobel_static_bg` instead of intensity direction |
| Bent/leaning person classified as object | Bent-person branch with spatial gate (must be near existing person track) |
| Smooth ADDED objects drop from tracking | Intensity-only persistence: reset `missed` if mean diff > threshold/2 |
| Adaptive edge threshold fails on heavy-tailed distributions | `τ_e = max(6, median(diff_edge))` — distribution-agnostic |

---

## Project Structure

```
Ugolini_Filippo_0001146279/
├── README.md
├── .gitignore
└── intrusion_detection_code/
    ├── main.py                          # Main detection pipeline
    ├── report_generator_v2.py           # Generates report figures from saved data
    ├── rilevamento-intrusioni-video.mp4 # Input surveillance video
    ├── report_images/                   # Figures used in the written report
    │   ├── 01_pipeline_completa.png
    │   ├── 02_illumination_analysis.png
    │   ├── 03_edge_comparison.png
    │   ├── 04_morphology_steps.png
    │   ├── 05_contour_refinement.png
    │   ├── 06_classification_features.png
    │   ├── 07_tracking_temporal.png
    │   ├── 08_removed_objects.png
    │   └── 09_summary_results.png
    ├── Project_report_Ugolini_Filippo.pdf
    ├── Change Detection Project.pdf     # Assignment specification
    └── output/                          # Generated on run (git-ignored)
        ├── debug_labeled_contours.mp4   # Annotated output video
        ├── output_blobs.csv             # Per-frame detection records
        └── analysis_data/               # Per-frame debug images & JSON logs
```

---

## Requirements

Python 3.8+

```bash
pip install opencv-python numpy matplotlib pandas
```

---

## Usage

### 1. Run the detection system

```bash
cd intrusion_detection_code
python main.py
```

The algorithm:
1. Initializes the background model from the first 100 frames (median)
2. Processes all remaining frames, detecting PERSON / ADDED / REMOVED events
3. Saves results to `output/`

**Output:**
- `output/debug_labeled_contours.mp4` — annotated video with colored bounding boxes
- `output/output_blobs.csv` — CSV with per-frame blob features and classifications
- `output/analysis_data/` — per-frame debug images (10 intermediate masks per frame) and JSON logs

### 2. Generate report figures

```bash
python report_generator_v2.py
```

Reads the data saved by `main.py` and produces 9 analysis figures in `report_images/`.

---

## Configuration

All parameters are in the `Config` dataclass at the top of `main.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_frames` | 100 | Frames used to build the initial background model |
| `alpha` | 0.08 | Background learning rate |
| `diff_threshold` | 18 | Minimum intensity change to enter foreground |
| `edge_threshold_min` | 6 | Minimum edge change (adaptive median floor) |
| `min_area_person` | 500 | Minimum blob area for person pass (px) |
| `min_area_obj` | 400 | Minimum blob area for object pass (px) |
| `object_confirm_frames` | 15 | Frames before an ADDED track is confirmed |
| `removed_obj_confirm_frames` | 25 | Frames before a REMOVED track is confirmed |
| `removal_excl_threshold` | 30 | Edge-difference threshold for removal-direction exclusion |

---

## Debug Masks (per-frame)

Each frame in `analysis_frames` saves 10 intermediate images:

| File | Content |
|------|---------|
| `01_original.png` | Raw BGR frame |
| `02_gray.png` | Grayscale |
| `03_clahe.png` | After CLAHE enhancement |
| `04_blur.png` | After Gaussian blur |
| `05_diff_intensity.png` | `\|bg - frame\|` intensity difference |
| `06_diff_edge.png` | `\|Sobel(bg) - Sobel(frame)\|` edge difference |
| `07_foreground_raw.png` | Raw foreground mask (intensity AND edge) |
| `08_fg_person.png` | PASS 1 intermediate (all blobs after person morphology) |
| `09_fg_objects.png` | PASS 2 output — ADDED object candidates |
| `10_result.png` | Final annotated frame |
