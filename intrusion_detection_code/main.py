"""
Intrusion Detection Code 
====================================================================

Output:
- Video : output/debug_labeled_contours.mp4
- CSV : output/output_blobs.csv
- Datas: output/analysis_data/ (per report)
"""

import os
import csv
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    # I/O
    video_path: str = "rilevamento-intrusioni-video.mp4"
    out_dir: str = "output"
    out_csv_path: str = "output/output_blobs.csv"
    out_video_path: str = "output/debug_labeled_contours.mp4"
    analysis_dir: str = "output/analysis_data"
    save_debug_video: bool = True
    window_name: str = "Change Detection - Improved"
    
    # Background
    init_frames: int = 100
    alpha: float = 0.08
    alpha_fast_factor: float = 8.0   # multiplier for background update in removed-object regions
    
    # Pre-processing
    blur_ksize: int = 5
    use_clahe: bool = True
    clahe_clip: float = 1.5          # reduced from 2.0 → less tile-border artefacts
    clahe_grid: Tuple[int, int] = (8, 8)
    
    # Change detection thresholds
    diff_threshold: int = 18
    # Adaptive edge threshold: tau_e = max(edge_threshold_min, median(diff_edge))
    # Physical meaning: "an edge difference is significant if it exceeds the median edge activity
    # of the current frame" — the dividing line between below-average and above-average structural
    # change. Distribution-agnostic: works correctly for the heavy-tailed (Laplacian/exponential)
    # distribution that Sobel-difference maps follow in textured scenes.
    # For an exponential distribution with mean μ: median = μ·ln(2) ≈ 0.693·μ
    # E.g. at frame 120 (mean=25.2): median ≈ 17.5 — vs the broken σ-based value of 82.1
    edge_threshold_min: int = 6      # absolute floor for fully static scenes
    
    # Morphology PERSON
    min_area_person: int = 500
    close_kernel_person: int = 9        # reverted 7→9: required to fill hollow outlines in fg_person
    open_kernel_person: int = 5
    close_iter_person: int = 5          # reverted 3→5: effective closing radius ~33px needed
    open_iter_person: int = 4
    
    # Morphology OBJECTS
    min_area_obj: int = 400
    close_kernel_obj: int = 11
    open_kernel_obj: int = 5
    close_iter_obj: int = 3
    open_iter_obj: int = 1
    
    # Contour refinement
    use_contour_smoothing: bool = True
    contour_epsilon_factor: float = 0.004
    use_convex_hull_person: bool = False
    
    # Person Classification
    person_min_area: int = 1200
    person_min_height: int = 80
    
    # Object Filters
    obj_min_solidity: float = 0.35
    obj_min_extent: float = 0.15
    border_reject_px: int = 12
    
    # Tracking
    track_iou_threshold: float = 0.2
    track_max_dist: float = 80.0     # centroid-distance fallback when IoU < threshold
    max_missed: int = 20
    # Confirmed ADDED ("other", not removed) tracks persist longer when temporarily invisible
    # (e.g. person occludes the object). 60 frames ≈ 5 s at 12fps — enough for the person to
    # move away. During this time the bg-blocking zone (update_mask=0) also stays active,
    # preventing the background from absorbing the object footprint.
    object_max_missed_confirmed: int = 60
    person_dilate_px: int = 18
    # Spatial gate for the bent-person rule (ar < 1.0 branch of classify_person).
    # A blob passing only via the bent-person rule is accepted as a person only if
    # its centroid is within (person_bent_max_dist_factor × person_major_dim) pixels
    # of any existing person track. Objects (bags, boxes) pass the same shape criteria
    # but are spatially isolated from the person body.
    person_bent_max_dist_factor: float = 1.5

    # Object confirm
    object_confirm_frames: int = 15
    
    # Removal object detection
    detect_removed_objects: bool = True
    removed_obj_threshold: int = 20
    # Separate threshold for PASS 2 removal-direction exclusion.
    # removed_obj_threshold=20 catches weak removals in PASS 3; removal_excl_threshold=30 is
    # more selective in PASS 2 to avoid filtering ADDED objects on textured desks where
    # sobel_bg naturally exceeds sobel_frame by >20 without any object being removed.
    removal_excl_threshold: int = 30
    removed_obj_min_area: int = 400   # raised from 250 — desk-change noise blobs are typically <300px
    removed_obj_confirm_frames: int = 25
    
    # Display
    display_scale: float = 2.0

    # Analysis: histograms for threshold selection
    save_threshold_hists: bool = True
    hist_stride: int = 10          # save every 10 frames
    hist_max_frames: int = 10     
    
    analysis_frames: List[int] = None
    
    def __post_init__(self):
        if self.analysis_frames is None:
            # Frame: starting, foreground, object removal, end
            self.analysis_frames = [120, 180, 250, 320, 360, 380, 390, 420]


# ============================================================
# CLASSE PER SALVARE DATI ANALISI
# ============================================================
class AnalysisLogger:
    """Salva dati intermedi per generare report."""
    
    def __init__(self, analysis_dir: str):
        self.analysis_dir = analysis_dir
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Storage data
        self.frame_data = {}
        self.tracking_data = []
        self.features_log = []
        self.background = None
        self.config = None
        
    def save_background(self, bg: np.ndarray):
        """Salva background."""
        self.background = bg
        cv2.imwrite(f"{self.analysis_dir}/background.png", bg)
        
    def save_config(self, cfg: Config):
        """Salva configurazione."""
        self.config = cfg
        with open(f"{self.analysis_dir}/config.json", "w") as f:
            # Converti dataclass a dict
            cfg_dict = asdict(cfg)
            json.dump(cfg_dict, f, indent=2)
    
    def log_frame_processing(self, frame_idx: int, frame: np.ndarray,
                            gray: np.ndarray, gray_clahe: np.ndarray,
                            gray_blur: np.ndarray, bg_blur: np.ndarray,
                            diff: np.ndarray, diff_edge: np.ndarray,
                            fg: np.ndarray, fg_person: np.ndarray,
                            fg_obj: np.ndarray, result: np.ndarray):
        """Salva dati processing di un frame."""
        
        # Salva immagini intermedie
        frame_dir = f"{self.analysis_dir}/frame_{frame_idx:04d}"
        os.makedirs(frame_dir, exist_ok=True)
        
        cv2.imwrite(f"{frame_dir}/01_original.png", frame)
        cv2.imwrite(f"{frame_dir}/02_gray.png", gray)
        cv2.imwrite(f"{frame_dir}/03_clahe.png", gray_clahe)
        cv2.imwrite(f"{frame_dir}/04_blur.png", gray_blur)
        cv2.imwrite(f"{frame_dir}/05_diff_intensity.png", diff)
        cv2.imwrite(f"{frame_dir}/06_diff_edge.png", diff_edge)
        cv2.imwrite(f"{frame_dir}/07_foreground_raw.png", fg)
        cv2.imwrite(f"{frame_dir}/08_fg_person.png", fg_person)
        cv2.imwrite(f"{frame_dir}/09_fg_objects.png", fg_obj)
        cv2.imwrite(f"{frame_dir}/10_result.png", result)
        
        # Salva istogrammi
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_clahe = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256])
        
        self.frame_data[frame_idx] = {
            'histograms': {
                'gray': hist_gray.flatten().tolist(),
                'clahe': hist_clahe.flatten().tolist(),
            }
        }
        
    def log_features(self, frame_idx: int, track_id: int, 
                    features: Dict, classification: str):
        """Log features estratte."""
        self.features_log.append({
            'frame': frame_idx,
            'track_id': track_id,
            'classification': classification,
            'area': features['area'],
            'perimeter': features['perimeter'],
            'bbox_w': features['bbox_w'],
            'bbox_h': features['bbox_h'],
            'cx': features['cx'],
            'cy': features['cy'],
            'extent': features['extent'],
            'solidity': features['solidity'],
            'aspect_ratio': features['aspect_ratio'],
        })
    
    def log_tracking(self, frame_idx: int, tracks: List):
        """Log stato tracking."""
        for track in tracks:
            self.tracking_data.append({
                'frame': frame_idx,
                'track_id': track.tid,
                'classification': track.cls,
                'cx': track.feats['cx'],
                'cy': track.feats['cy'],
                'is_confirmed': track.is_confirmed,
                'seen_count': track.seen_count,
            })
    
    def save_all(self):
        """Salva tutti i dati aggregati."""
        # Features
        with open(f"{self.analysis_dir}/features.json", "w") as f:
            json.dump(self.features_log, f, indent=2)
        
        # Tracking
        with open(f"{self.analysis_dir}/tracking.json", "w") as f:
            json.dump(self.tracking_data, f, indent=2)
        
        # Frame data
        with open(f"{self.analysis_dir}/frame_data.json", "w") as f:
            json.dump(self.frame_data, f, indent=2)
        
        print(f"\n[ANALYSIS] Saving datas in: {self.analysis_dir}/")
        print(f"  - background.png")
        print(f"  - config.json")
        print(f"  - features.json ({len(self.features_log)} records)")
        print(f"  - tracking.json ({len(self.tracking_data)} records)")
        print(f"  - frame_data.json ({len(self.frame_data)} frames)")
        print(f"  - frame_XXXX/ ({len(self.frame_data)} cartelle)")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def area_opening(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def compute_edges_sobel(gray: np.ndarray) -> np.ndarray:
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def refine_contour(contour: np.ndarray, cfg: Config, use_hull: bool = False,
                   img_w: int = None, img_h: int = None) -> np.ndarray:
    if len(contour) < 3:
        return contour
    refined = contour
    if cfg.use_contour_smoothing:
        epsilon = cfg.contour_epsilon_factor * cv2.arcLength(contour, True)
        refined = cv2.approxPolyDP(refined, epsilon, True)
    if use_hull:
        refined = cv2.convexHull(refined)
    # Clip contour points to image bounds to prevent erratic drawing near frame borders
    if img_w is not None and img_h is not None and len(refined) > 0:
        refined = np.clip(refined, [[[ 0,  0]]], [[[img_w - 1, img_h - 1]]])
    return refined

def compute_blob_features(mask: np.ndarray, cfg: Config, is_person: bool = False,
                          img_w: int = None, img_h: int = None) -> Optional[Dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    cnt_refined = refine_contour(cnt, cfg, use_hull=is_person and cfg.use_convex_hull_person,
                                 img_w=img_w, img_h=img_h)
    
    area = cv2.contourArea(cnt_refined)
    perimeter = cv2.arcLength(cnt_refined, True)
    x, y, w, h = cv2.boundingRect(cnt_refined)
    cx = x + w / 2.0
    cy = y + h / 2.0
    extent = area / (w * h) if w * h > 0 else 0
    hull = cv2.convexHull(cnt_refined)
    hull_area = cv2.contourArea(hull) if hull is not None else area
    solidity = area / hull_area if hull_area > 1e-6 else 0
    aspect_ratio = h / w if w > 0 else 0
    
    return {
        "area": area, "perimeter": perimeter,
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        "cx": cx, "cy": cy, "extent": extent,
        "solidity": solidity, "aspect_ratio": aspect_ratio,
        "contour": cnt_refined,
    }

def classify_person(features: Dict, cfg: Config) -> str:
    if not features:
        return "other"
    area = features["area"]
    h = features["bbox_h"]
    w = features["bbox_w"]
    sol = features["solidity"]
    ext = features["extent"]
    ar = features["aspect_ratio"]   # = h / w

    # ── Fast-pass: unambiguous upright person ──────────────────────────────────
    if area >= 3000 and h >= 80 and w >= 50 and ar >= 1.0:
        return "person"

    # ── Hard reject: too small in all dimensions ───────────────────────────────
    if area < 1200 or (h < 50 and w < 50):
        return "other"

    # ── Shape too sparse (neither standing nor bent persons are scattered) ─────
    if sol < 0.12 or ext < 0.08:
        return "other"

    # ── Standard upright person (taller than wide) ────────────────────────────
    if ar >= 1.0:
        big_enough = area >= cfg.person_min_area and h >= cfg.person_min_height
        good_shape = (sol >= 0.15 and ext >= 0.10 and ar <= 4.0)
        return "person" if (big_enough and good_shape) else "other"

    # ── Bent / leaning person (wider than tall) ───────────────────────────────
    # When a person bends over a desk, h/w (ar) drops below 1.0 but the blob
    # remains large and compact. Without this branch the old `if w > h: return
    # "other"` silently dropped the bent person from person_mask, which caused:
    #   (a) the person track to accumulate missed frames → unstable tracking (P1)
    #   (b) the bent-person blob to enter obj_dets → false ADDED label (P5)
    # Thresholds chosen conservatively: area≥2500 (desk objects are smaller),
    # solidity≥0.20 (person body is compact), extent≥0.12, ar≥0.35 (avoids
    # pure horizontal blobs wider than 3× their height, e.g. a shelf).
    if ar >= 0.35 and area >= 2500 and sol >= 0.20 and ext >= 0.12:
        return "person"

    return "other"

def reject_object(features: Dict, cfg: Config, img_w: int, img_h: int) -> bool:
    x, y, w, h = features["bbox_x"], features["bbox_y"], features["bbox_w"], features["bbox_h"]
    if w < 10 or h < 10:
        return True
    b = cfg.border_reject_px
    if x <= b or y <= b or (x + w) >= (img_w - b) or (y + h) >= (img_h - b):
        return True
    if features["solidity"] < cfg.obj_min_solidity or features["extent"] < cfg.obj_min_extent:
        return True
    return False

def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

class Track:
    def __init__(self, tid: int, feats: Dict, cls: str):
        self.tid = tid
        self.cls = cls
        self.feats = feats
        self.missed = 0
        self.seen_count = 1
        self.is_removed = feats.get("_is_removed", False)
        self.first_cx = feats["cx"]
        self.first_cy = feats["cy"]
        self.static_count = 0
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        if not self.feats:
            return (0, 0, 0, 0)
        return (self.feats["bbox_x"], self.feats["bbox_y"], 
                self.feats["bbox_w"], self.feats["bbox_h"])
    
    def update(self, feats: Dict):
        if self.cls == "person":
            feats["_is_removed"] = False
        # If a confirmed REMOVED track is matched with an ADDED detection (e.g. a new object
        # placed at the same location as a previously removed one), reclassify as not-removed
        # so the track transitions from "REMOVED" to "ADDED" label in the output.
        if self.is_removed and not feats.get("_is_removed", False):
            self.is_removed = False
        self.feats = feats
        self.missed = 0
        self.seen_count += 1
        if self.cls == "person":
            dx = abs(feats["cx"] - self.first_cx)
            dy = abs(feats["cy"] - self.first_cy)
            if (dx**2 + dy**2)**0.5 < 3.0:
                self.static_count += 1
            else:
                self.static_count = 0
                self.first_cx = feats["cx"]
                self.first_cy = feats["cy"]
    
    @property
    def is_confirmed(self) -> bool:
        if self.cls == "person":
            return True
        # REMOVED needs 20 frames (was 12) to filter out noise sources that persist briefly:
        # chair displacement, person shadow, CLAHE variance. True removed objects are permanent
        # (static_bg is frozen) so raising the threshold doesn't affect them, only eliminates FP.
        threshold = 20 if self.is_removed else 15
        return self.seen_count >= threshold

def update_tracks(tracks: List[Track], detections: List[Dict],
                  cfg: Config, next_id: int) -> Tuple[List[Track], int]:
    unmatched = list(range(len(detections)))
    for track in tracks:
        best_iou = 0.0
        best_idx = -1
        for i in unmatched:
            det = detections[i]
            det_bbox = (det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"])
            iou_val = iou(track.bbox, det_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        matched = best_iou >= cfg.track_iou_threshold

        # Centroid-distance fallback: when IoU is too low (e.g. fast motion, partial overlap)
        # but a detection's centroid is within track_max_dist pixels, still associate it.
        # This prevents unnecessary identity switches during quick movement.
        #
        # P2 fix: disable the centroid fallback for CONFIRMED static added objects.
        # A confirmed non-removed "other" track is a stationary object — it should not
        # move. Allowing centroid fallback caused track #8 to absorb a leg/foot detection
        # at cx=141 (42px jump from the object's true cx=103), corrupting t.feats.bbox
        # and shifting the background-protection zone away from the real object.
        # With the fallback disabled for these tracks, the detection at the wrong position
        # remains unmatched (spawns a short-lived new track) while the confirmed track
        # correctly increments missed and retains its true bbox for bg protection.
        is_static_confirmed = (track.is_confirmed and
                               track.cls == "other" and
                               not track.is_removed)
        if not matched and track.feats and not is_static_confirmed:
            best_dist = float('inf')
            best_dist_idx = -1
            for i in unmatched:
                det = detections[i]
                dx = track.feats["cx"] - det["cx"]
                dy = track.feats["cy"] - det["cy"]
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_dist_idx = i
            if best_dist < cfg.track_max_dist:
                best_idx = best_dist_idx
                matched = True

        if matched and best_idx >= 0:
            if track.cls == "person":
                detections[best_idx]["_instant_cls"] = "person"
                detections[best_idx]["_is_removed"] = False
            track.update(detections[best_idx])
            unmatched.remove(best_idx)
        else:
            track.missed += 1
    
    for i in unmatched:
        det = detections[i]
        cls = det.get("_instant_cls", "other")
        if cls == "person":
            det["_is_removed"] = False
        tracks.append(Track(next_id, det, cls))
        next_id += 1
    
    tracks[:] = [t for t in tracks if
                 t.missed <= (cfg.object_max_missed_confirmed
                              if t.is_confirmed and t.cls == "other" and not t.is_removed
                              else cfg.max_missed)]
    return tracks, next_id

def detect_removed_objects(static_bg: np.ndarray, frame: np.ndarray,
                          person_mask: np.ndarray, cfg: Config,
                          sobel_bg: np.ndarray = None, sobel_frame: np.ndarray = None,
                          img_w: int = None, img_h: int = None) -> List[Dict]:
    if not cfg.detect_removed_objects:
        return []

    # REMOVAL DETECTION using the FROZEN static_bg reference.
    #
    # Root-cause fix (P1): the adaptive bg absorbs the removal signal within ~70 frames
    # while the person is at the desk (the desk area is not in fg_union → bg updates freely).
    # By the time the person walks away, diff ≈ 0 → no removal signal.
    #
    # Solution: use static_bg (frozen at init, never updated) as the reference.
    # static_bg always has the original desk object → diff stays large permanently
    # even after the adaptive bg has fully adapted to the missing object.
    #
    # Edge-direction filter still uses sobel_bg (= sobel of static_bg):
    # "bg had edges, frame doesn't" → object is absent from current frame.
    # For an ADDED object the reverse is true (frame has more edges) → not detected here.
    if sobel_bg is None:
        sobel_bg = compute_edges_sobel(static_bg)
    if sobel_frame is None:
        sobel_frame = compute_edges_sobel(frame)

    diff_intensity = cv2.absdiff(static_bg, frame)
    _, intensity_mask = cv2.threshold(diff_intensity, cfg.removed_obj_threshold, 255, cv2.THRESH_BINARY)

    edge_removal_dir = cv2.subtract(sobel_bg, sobel_frame)   # >0 only where bg had more edges
    _, edge_dir_mask = cv2.threshold(edge_removal_dir, cfg.removed_obj_threshold, 255, cv2.THRESH_BINARY)

    removed_mask = cv2.bitwise_and(intensity_mask, edge_dir_mask)

    if np.any(person_mask > 0):
        # Reduced exclusion zone: was (45,45) iter=3 → ~67 px, then (15,15) iter=2 → ~20 px.
        # Now (7,7) iter=1 → ~5 px: excludes only immediate edge/shadow pixels of the body,
        # not nearby desk objects. This allows removal detection to trigger earlier — while the
        # person is still in the frame but has already moved a few pixels away from the object.
        # Any extra FP near the person is filtered by the raised confirmation threshold (20 frames).
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        person_exclusion = cv2.dilate(person_mask, k, iterations=1)
        removed_mask = cv2.bitwise_and(removed_mask, cv2.bitwise_not(person_exclusion))

    # P3 fix: reduced morphological aggressiveness to preserve spatial accuracy.
    # Old parameters: close (9,9)×3 → effective ~27 px radius. This merged adjacent
    # removed objects on the desk into one large blob whose bounding box spanned both,
    # making the REMOVED label appear between the objects rather than on them.
    # New parameters: close (7,7)×2 → effective ~14 px radius; open (5,5)×1 → noise
    # removal with minimal blob shrinkage. Objects closer than ~14 px may still merge
    # (intentional — they would share a bounding box at that scale anyway).
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    removed_mask = cv2.morphologyEx(removed_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    removed_mask = cv2.morphologyEx(removed_mask, cv2.MORPH_OPEN, k_open, iterations=1)
    removed_mask = area_opening(removed_mask, cfg.removed_obj_min_area)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(removed_mask, connectivity=8)

    removed_objs = []
    for i in range(1, num):
        area_pix = stats[i, cv2.CC_STAT_AREA]
        # Relaxed upper area limit: was 2000 → desk objects can be larger
        if area_pix < cfg.removed_obj_min_area or area_pix > 5000:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        feats = compute_blob_features(comp, cfg, is_person=False, img_w=img_w, img_h=img_h)
        if not feats:
            continue
        h = feats["bbox_h"]
        ar = feats["aspect_ratio"]
        # Relaxed shape filters: was ar>=1.2 h>80 and h>90
        if ar >= 2.0 and h > 120:
            continue
        if h > 180:
            continue
        # Raised thresholds (was solidity<0.08, extent<0.04) to filter noisy/fragmented blobs.
        # Real desk objects are compact; shadow/noise artifacts are typically sparse.
        if feats["solidity"] < 0.15 or feats["extent"] < 0.08:
            continue
        feats["_instant_cls"] = "other"
        feats["_is_removed"] = True
        removed_objs.append(feats)

    return removed_objs

def save_threshold_histograms(diff: np.ndarray,
                              diff_edge: np.ndarray,
                              cfg: Config,
                              adaptive_edge_thr: float,
                              frame_idx: int,
                              saved_counter: int) -> int:
    """
    Save histograms of diff and diff_edge to justify threshold selection.
    adaptive_edge_thr is the per-frame sigma-based edge threshold used for fg computation.
    Returns the updated counter of saved histogram frames.
    """
    if not cfg.save_threshold_hists:
        return saved_counter
    if saved_counter >= cfg.hist_max_frames:
        return saved_counter
    if frame_idx % cfg.hist_stride != 0:
        return saved_counter

    out_dir = os.path.join(cfg.analysis_dir, "threshold_hists")
    os.makedirs(out_dir, exist_ok=True)

    # Histogram diff (intensity difference)
    diff_hist, _ = np.histogram(diff.ravel(), bins=256, range=(0, 255))

    plt.figure()
    plt.plot(diff_hist)
    plt.axvline(x=cfg.diff_threshold, linestyle="--", label=f"tau_d = {cfg.diff_threshold}")
    plt.title(f"Histogram of intensity difference (frame {frame_idx})")
    plt.xlabel("Difference value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_diff_f{frame_idx:05d}.png"), dpi=150)
    plt.close()

    # Histogram diff_edge with median-based adaptive threshold annotation
    edge_hist, _ = np.histogram(diff_edge.ravel(), bins=256, range=(0, 255))
    e_median = float(np.median(diff_edge))
    e_mean   = float(np.mean(diff_edge))

    plt.figure()
    plt.plot(edge_hist, label="diff_edge histogram")
    plt.axvline(x=adaptive_edge_thr, linestyle="--", color="red",
                label=f"tau_e = {adaptive_edge_thr:.1f}  (median-based)")
    plt.axvline(x=e_median, linestyle=":", color="orange", alpha=0.8,
                label=f"median = {e_median:.1f}")
    plt.axvline(x=e_mean, linestyle=":", color="blue", alpha=0.5,
                label=f"mean = {e_mean:.1f}")
    plt.title(f"Histogram of edge difference (frame {frame_idx})\n"
              f"Adaptive threshold = median(diff_edge) = {adaptive_edge_thr:.1f}\n"
              f"Distribution is heavy-tailed: mean+1.5σ would give {e_mean + 1.5*float(np.std(diff_edge)):.1f} (too high)")
    plt.xlabel("Edge difference value")
    plt.ylabel("Count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_edge_f{frame_idx:05d}.png"), dpi=150)
    plt.close()

    return saved_counter + 1


def save_clahe_histograms(gray, clahe_img, frame_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    hist_before, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_after, _ = np.histogram(clahe_img.ravel(), bins=256, range=(0, 256))
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(hist_before)
    plt.title("Histogram before CLAHE")
    plt.xlabel("Gray level")
    plt.ylabel("Pixel count")

    plt.subplot(1,2,2)
    plt.plot(hist_after)
    plt.title("Histogram after CLAHE")
    plt.xlabel("Gray level")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"clahe_hist_frame_{frame_idx}.png"), dpi=150)
    plt.close()
# ============================================================
# MAIN PROCESSING
# ============================================================
def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.analysis_dir, exist_ok=True)
    
    # Logger initialization
    logger = AnalysisLogger(cfg.analysis_dir)
    logger.save_config(cfg)
    
    # Open the video
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_ms = int(1000.0 / fps)
    
    # Setup CLAHE
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, 
                            tileGridSize=cfg.clahe_grid) if cfg.use_clahe else None
    
    # Initilaization background
    print("[INFO] Initializing background...")
    bg_frames = []
    for _ in range(cfg.init_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if clahe:
            gray = clahe.apply(gray)   # single application (was applied twice before — bug B1)
        gray = gaussian_blur(gray, cfg.blur_ksize).astype(np.float32)
        bg_frames.append(gray)

    # Store background as float32 to avoid uint8 quantization drift on repeated updates
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.float32)
    logger.save_background(bg.astype(np.uint8))

    # Frozen reference background for removed-object detection (P1 fix).
    # The adaptive bg absorbs the removal signal within ~70 frames while the person
    # is at the desk. static_bg is never updated → diff vs removed objects stays large
    # for the entire video, enabling reliable detection long after the adaptive bg drifts.
    static_bg = bg.copy()
    static_bg_uint8 = static_bg.astype(np.uint8)
    static_bg_f = gaussian_blur(static_bg_uint8, cfg.blur_ksize)
    sobel_static_bg_f = compute_edges_sobel(static_bg_f)

    # Morphologic Kernels 
    k_close_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (cfg.close_kernel_person, cfg.close_kernel_person))
    k_open_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (cfg.open_kernel_person, cfg.open_kernel_person))
    k_close_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (cfg.close_kernel_obj, cfg.close_kernel_obj))
    k_open_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (cfg.open_kernel_obj, cfg.open_kernel_obj))
    
    # Setup output
    writer = None
    if cfg.save_debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg.out_video_path, fourcc, fps, (W, H))
    
    csv_file = open(cfg.out_csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "frame_index", "num_objects", "object_id", "classification",
        "area", "perimeter", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "cx", "cy", "extent", "solidity", "aspect_ratio",
        "is_confirmed", "is_removed", "seen_count"
    ])
    csv_writer.writeheader()
    
    # Tracking
    tracks = []
    next_track_id = 1
    hist_saved = 0
    frame_idx = cfg.init_frames
    
    print("[INFO] Processing frames (saving analysis data)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_raw = gray.copy()
        if clahe:
            gray = clahe.apply(gray)
        gray_clahe = gray.copy()           # save CLAHE result before blur (fixes NameError B2)
        gray_f = gaussian_blur(gray, cfg.blur_ksize)
        # bg is stored as float32; cast to uint8 before pixel-level cv2 operations
        bg_uint8 = bg.astype(np.uint8)
        bg_f = gaussian_blur(bg_uint8, cfg.blur_ksize)

        if frame_idx in [50, 150, 250, 300, 380]:  # scegli frame rappresentativi
            save_clahe_histograms(gray_raw, gray_clahe, frame_idx, "output/analysis_data/clahe_hist")
            print('Saving..')
            

        # Change detection
        diff = cv2.absdiff(bg_f, gray_f)
        # Compute Sobel maps once; reused for diff_edge, fg_o removal-direction exclusion,
        # and detect_removed_objects — avoids redundant per-frame Sobel computation.
        sobel_bg_f   = compute_edges_sobel(bg_f)
        sobel_gray_f = compute_edges_sobel(gray_f)
        diff_edge = cv2.absdiff(sobel_bg_f, sobel_gray_f)

        # Adaptive edge threshold: median of the per-frame Sobel-difference map.
        # Root cause of previous failure: mean+1.5σ gave τ_e≈82 on this scene because the
        # diff_edge distribution is heavy-tailed (Laplacian/exponential), not Gaussian.
        # The σ-based formula placed the threshold far up the tail, blocking person-interior pixels
        # and making fg_person completely empty.
        # Median is distribution-agnostic: it marks the 50th percentile of edge activity,
        # adapting to scene texture without being skewed by the heavy tail.
        # For exponential(μ=25.2): median ≈ 17.5 — in the same range as the old fixed τ_e=12.
        adaptive_edge_thr = max(cfg.edge_threshold_min, float(np.median(diff_edge)))

        hist_saved = save_threshold_histograms(diff, diff_edge, cfg, adaptive_edge_thr,
                                               frame_idx, hist_saved)

        fg = ((diff > cfg.diff_threshold) &
              (diff_edge > adaptive_edge_thr)).astype(np.uint8) * 255
        
        # PASS 1: PERSON
        fg_p = area_opening(fg, cfg.min_area_person)
        fg_p = cv2.morphologyEx(fg_p, cv2.MORPH_CLOSE, k_close_p, iterations=cfg.close_iter_person)
        fg_p = cv2.morphologyEx(fg_p, cv2.MORPH_OPEN, k_open_p, iterations=cfg.open_iter_person)
        
        num_p, labels_p, stats_p, _ = cv2.connectedComponentsWithStats(fg_p, connectivity=8)
        
        person_mask = np.zeros((H, W), dtype=np.uint8)
        person_dets = []

        # Collect person track positions from the previous frame to anchor the
        # bent-person disambiguation check inside the classification loop below.
        person_track_bboxes = [(t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
                               for t in tracks if t.cls == "person"]

        for i in range(1, num_p):
            if stats_p[i, cv2.CC_STAT_AREA] < cfg.min_area_person:
                continue
            comp = (labels_p == i).astype(np.uint8) * 255
            feats = compute_blob_features(comp, cfg, is_person=True, img_w=W, img_h=H)
            if not feats:
                continue
            if classify_person(feats, cfg) == "person":
                # Spatial gate for the bent-person branch (ar < 1.0).
                # Blobs with ar < 1.0 pass classify_person via the bent-person rule,
                # which compact objects (bags, boxes) can also satisfy. Accept them as
                # person only when their centroid is close to a known person track.
                # If no person track exists yet (initialization), the check is skipped
                # and standard classification applies.
                if feats["aspect_ratio"] < 1.0 and person_track_bboxes:
                    fcx = feats["bbox_x"] + feats["bbox_w"] // 2
                    fcy = feats["bbox_y"] + feats["bbox_h"] // 2
                    close_to_any = False
                    for bx, by, bw, bh in person_track_bboxes:
                        pcx = bx + bw // 2
                        pcy = by + bh // 2
                        ref_dim = max(bw, bh)
                        if ((fcx - pcx) ** 2 + (fcy - pcy) ** 2) ** 0.5 <= cfg.person_bent_max_dist_factor * ref_dim:
                            close_to_any = True
                            break
                    if not close_to_any:
                        continue  # demote: pixels remain in fg → appear in fg_o → ADDED
                person_mask[labels_p == i] = 255
                feats["_instant_cls"] = "person"
                person_dets.append(feats)
        
        # Person dilation
        if cfg.person_dilate_px > 0:
            k = cfg.person_dilate_px if cfg.person_dilate_px % 2 == 1 else cfg.person_dilate_px + 1
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            person_mask = cv2.dilate(person_mask, kd, iterations=1)
        
        # PASS 2: OBJECTS (additions only)
        fg_o = cv2.bitwise_and(fg, cv2.bitwise_not(person_mask))

        # Exclude removal-direction pixels from fg_o using EDGE-DIRECTION from frozen static_bg.
        #
        # Root cause of previous failure (intensity-direction approach):
        # The assumption "ADDED objects make gray_f > bg_f" is WRONG for dark objects on a
        # light background. A dark trash can on a light floor satisfies bg_f > gray_f, so the
        # intensity-direction filter incorrectly excluded the ADDED object from fg_o, making
        # 09_fg_objects always empty and the object invisible to PASS 2.
        #
        # Physical correctness of edge-direction with frozen static_bg:
        # - REMOVED object: static_bg had edges (the object), current frame is smooth (absent)
        #   → sobel_static_bg > sobel_frame → excluded from fg_o ✓
        # - ADDED dark object on smooth floor: static_bg was flat (no edges), current frame
        #   has edges at the new object boundary → sobel_frame > sobel_static_bg
        #   → NOT excluded → ADDED object reaches fg_o ✓
        # - ADDED object on textured background (edge case): sobel_static_bg may exceed
        #   sobel_frame in the object interior; raise removal_excl_threshold if regression seen.
        edge_removal_static = np.clip(
            sobel_static_bg_f.astype(np.int32) - sobel_gray_f.astype(np.int32),
            0, 255
        ).astype(np.uint8)
        _, removal_excl = cv2.threshold(edge_removal_static, cfg.removal_excl_threshold,
                                        255, cv2.THRESH_BINARY)
        fg_o = cv2.bitwise_and(fg_o, cv2.bitwise_not(removal_excl))

        fg_o = area_opening(fg_o, cfg.min_area_obj)
        fg_o = cv2.morphologyEx(fg_o, cv2.MORPH_CLOSE, k_close_o, iterations=cfg.close_iter_obj)
        fg_o = cv2.morphologyEx(fg_o, cv2.MORPH_OPEN, k_open_o, iterations=cfg.open_iter_obj)
        
        num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(fg_o, connectivity=8)
        
        obj_dets = []
        for i in range(1, num_o):
            if stats_o[i, cv2.CC_STAT_AREA] < cfg.min_area_obj:
                continue
            comp = (labels_o == i).astype(np.uint8) * 255
            feats = compute_blob_features(comp, cfg, is_person=False, img_w=W, img_h=H)
            if not feats:
                continue
            if reject_object(feats, cfg, W, H):
                continue
            feats["_instant_cls"] = "other"
            obj_dets.append(feats)
        
        # PASS 3: REMOVED OBJECTS
        # Use static_bg_f (frozen) instead of bg_f (adaptive) so the removal signal
        # persists even after the adaptive background has adapted to the missing object.
        removed_dets = detect_removed_objects(static_bg_f, gray_f, person_mask, cfg,
                                              sobel_bg=sobel_static_bg_f, sobel_frame=sobel_gray_f,
                                              img_w=W, img_h=H)
        
        # Tracking
        all_dets = person_dets + obj_dets + removed_dets
        tracks, next_track_id = update_tracks(tracks, all_dets, cfg, next_track_id)

        # P4 fix: Intensity-only persistence check for confirmed ADDED tracks.
        #
        # Root cause: the main fg mask uses AND(diff>τ_d, diff_edge>τ_e). For smooth
        # added objects (book/bag on a smooth desk) the interior has low diff_edge →
        # fg_o is empty → no obj_det → track accumulates missed. This fix resets the
        # missed counter when the mean intensity diff in the confirmed object's bbox
        # still exceeds half the normal threshold, indicating the object is present
        # even though the stricter dual-condition pipeline did not fire.
        #
        # This works because the bg-blocking zone (update_mask=0, lines below) keeps
        # bg frozen in the confirmed object's region, so diff = |bg_frozen - frame|
        # remains large as long as the object is there — independent of diff_edge.
        _relax_thr = cfg.diff_threshold // 2  # 9 for default diff_threshold=18
        for _t in tracks:
            if (_t.is_confirmed and _t.cls == "other" and not _t.is_removed
                    and _t.missed > 0):
                _x, _y, _bw, _bh = _t.bbox
                _x0, _y0 = max(0, _x), max(0, _y)
                _x1, _y1 = min(W, _x + _bw), min(H, _y + _bh)
                _roi = diff[_y0:_y1, _x0:_x1]
                if _roi.size > 0 and float(np.mean(_roi)) > _relax_thr:
                    _t.missed = 0  # object still clearly present via intensity alone

        # Background update
        # fg_union covers person and added-object regions → block normal update there
        fg_union = cv2.bitwise_or(fg_p, fg_o)
        update_mask = np.ones_like(fg_union, dtype=np.uint8)
        update_mask[fg_union > 0] = 0
        update_mask[diff < 4] = 1   # always allow update in truly static areas

        # alpha_fast: accelerated learning rate for regions where an object was removed.
        # Physically: the background model must "forget" the removed object quickly so the
        # diff in that region returns to zero and false positives stop.
        alpha_fast = min(0.6, cfg.alpha * cfg.alpha_fast_factor)

        for t in tracks:
            # Fix A: block bg for ALL "other" tracks (tentative AND confirmed).
            # Previously `t.is_confirmed` was required; during the 15-frame tentative phase
            # bg updated freely → 1-(1-0.08)^15 ≈ 72% absorption → diff drops below threshold
            # before confirmation, breaking the detection feedback loop entirely.
            if t.cls == "other":
                x, y, w, h = t.bbox
                pad = 8
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
                if t.is_removed and t.seen_count > cfg.removed_obj_confirm_frames:
                    # BUG FIX (B3): removed objects must NOT block bg update.
                    # Instead, force a fast update so the background adapts to the empty region.
                    region = bg[y0:y1, x0:x1]
                    frame_region = gray_f[y0:y1, x0:x1].astype(np.float32)
                    bg[y0:y1, x0:x1] = (1.0 - alpha_fast) * region + alpha_fast * frame_region
                    update_mask[y0:y1, x0:x1] = 1   # also allow normal update pass below
                else:
                    # Added (placed) objects: block update to preserve their footprint in model
                    update_mask[y0:y1, x0:x1] = 0

        # bg stored as float32 to avoid uint8 quantization drift (B4)
        bg_new = (1.0 - cfg.alpha) * bg + cfg.alpha * gray_f.astype(np.float32)
        bg = np.where(update_mask > 0, bg_new, bg)   # stays float32
        
        # Output
        # P2 fix: show ALL confirmed tracks regardless of missed count.
        # A confirmed non-person track with missed > 0 means the object is temporarily
        # not visible (e.g. person occluding it), NOT that it disappeared.
        # The track is still alive (missed <= max_missed=20) → show its last known position.
        valid_tracks = [t for t in tracks if t.is_confirmed]
        valid_tracks = [t for t in valid_tracks if
                       t.cls != "person" or t.static_count < 20]
        
        # Draw
        vis = frame.copy()
        for t in valid_tracks:
            feats = t.feats
            x, y, w, h = feats["bbox_x"], feats["bbox_y"], feats["bbox_w"], feats["bbox_h"]
            
            if t.cls == "person":
                color = (0, 255, 0)
                label = f"#{t.tid} PERSON"
            elif t.is_removed:
                color = (255, 255, 0)
                label = f"#{t.tid} REMOVED"
            else:
                color = (255, 0, 0)
                label = f"#{t.tid} ADDED"
            
            cv2.drawContours(vis, [feats["contour"]], -1, color, 2)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, label, (x, max(0, y - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, label, (x, max(0, y - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # CSV
            csv_writer.writerow({
                "frame_index": frame_idx,
                "num_objects": len(valid_tracks),
                "object_id": t.tid,
                "classification": t.cls,
                "area": f"{feats['area']:.1f}",
                "perimeter": f"{feats['perimeter']:.1f}",
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "cx": f"{feats['cx']:.2f}",
                "cy": f"{feats['cy']:.2f}",
                "extent": f"{feats['extent']:.3f}",
                "solidity": f"{feats['solidity']:.3f}",
                "aspect_ratio": f"{feats['aspect_ratio']:.3f}",
                "is_confirmed": int(t.is_confirmed),
                "is_removed": int(t.is_removed),
                "seen_count": t.seen_count,
            })
            
            # Log features per report
            logger.log_features(frame_idx, t.tid, feats, t.cls)
        
        # Log tracking
        logger.log_tracking(frame_idx, valid_tracks)
        
        # Save the frame if it is on the list
        if frame_idx in cfg.analysis_frames:
            logger.log_frame_processing(
                frame_idx, frame, gray_raw, gray, gray_f, bg_f,
                diff, diff_edge, fg, fg_p, fg_o, vis
            )
        
        # Display
        vis_show = cv2.resize(vis, None, fx=cfg.display_scale, fy=cfg.display_scale,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(cfg.window_name, vis_show)
        
        if writer:
            writer.write(vis)
        
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Save the analysis data
    logger.save_all()
    
    # Cleanup
    csv_file.close()
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[OK] CSV: {cfg.out_csv_path}")
    if cfg.save_debug_video:
        print(f"[OK] Video: {cfg.out_video_path}")
    print(f"[OK] Analysis data: {cfg.analysis_dir}/")


if __name__ == "__main__":
    main()
