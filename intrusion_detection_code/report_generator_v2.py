"""
REPORT GENERATOR - Generate figure form the saved datas
=================================================

Prerequisit: Execute main.py in order to generate the datas

Output: folder 'report_images/' con tutte le figure
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List

# ============================================================
# Configuration
# ============================================================
ANALYSIS_DIR = "output/analysis_data"
OUTPUT_DIR = "report_images"
VIDEO_PATH = "rilevamento-intrusioni-video.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Loading data
# ============================================================

def load_analysis_data():
    """Upload all the save datas."""
    print("[INFO] Loading data analysis...")
    
    # Config
    with open(f"{ANALYSIS_DIR}/config.json", "r") as f:
        config = json.load(f)
    
    # Features
    with open(f"{ANALYSIS_DIR}/features.json", "r") as f:
        features = json.load(f)
    
    # Tracking
    with open(f"{ANALYSIS_DIR}/tracking.json", "r") as f:
        tracking = json.load(f)
    
    # Frame data
    with open(f"{ANALYSIS_DIR}/frame_data.json", "r") as f:
        frame_data = json.load(f)
    
    # Background
    background = cv2.imread(f"{ANALYSIS_DIR}/background.png", cv2.IMREAD_GRAYSCALE)
    
    print(f"  ✓ Config uploaded")
    print(f"  ✓ Features: {len(features)} records")
    print(f"  ✓ Tracking: {len(tracking)} records")
    print(f"  ✓ Frame data: {len(frame_data)} frames")
    print(f"  ✓ Background uploaded")
    
    return config, features, tracking, frame_data, background

# ============================================================
# FIGURE
# ============================================================

def figure_01_pipeline_completa(frame_idx: int):
    """Complete pipeline."""
    print(f"[1/9] Generating complete pipeline (frame {frame_idx})...")
    
    frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
    
    if not os.path.exists(frame_dir):
        print(f"  ⚠ Frame {frame_idx} not found, skipping...")
        return
    
    # load images
    images_data = [
        (cv2.imread(f"{frame_dir}/01_original.png"), 'RGB', '1. Frame Originale'),
        (cv2.imread(f"{frame_dir}/02_gray.png", 0), 'gray', '2. Grayscale'),
        (cv2.imread(f"{frame_dir}/03_clahe.png", 0), 'gray', '3. CLAHE'),
        (cv2.imread(f"{frame_dir}/04_blur.png", 0), 'gray', '4. Gaussian Blur'),
        (cv2.imread(f"{ANALYSIS_DIR}/background.png", 0), 'gray', '5. Background'),
        (cv2.imread(f"{frame_dir}/05_diff_intensity.png", 0), 'hot', '6. Diff Intensità'),
        (cv2.imread(f"{frame_dir}/06_diff_edge.png", 0), 'hot', '7. Diff Edge (Sobel)'),
        (cv2.imread(f"{frame_dir}/07_foreground_raw.png", 0), 'gray', '8. Foreground Raw'),
        (cv2.imread(f"{frame_dir}/08_fg_person.png", 0), 'gray', '9. After Morphology (Person)'),
        (cv2.imread(f"{frame_dir}/09_fg_objects.png", 0), 'gray', '10. After Morphology (Objects)'),
        (cv2.imread(f"{frame_dir}/10_result.png"), 'RGB', '11. Risultato Finale'),
    ]
    
    # Plot
    fig = plt.figure(figsize=(16, 12))
    rows, cols = 3, 4
    
    for idx, (img, cmap, title) in enumerate(images_data):
        if img is None:
            continue
        ax = plt.subplot(rows, cols, idx + 1)
        if cmap == 'RGB':
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap if cmap != 'gray' else 'gray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'Processin Pipeline complete (Frame {frame_idx})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_pipeline_complete.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 01_pipeline_complete.png")


def figure_02_illumination_analysis(frame_idx: int, frame_data: Dict):
    """Light alalysis with CLAHE."""
    print(f"[2/9] Generating light analysis ...")
    
    frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
    
    if not os.path.exists(frame_dir):
        print(f" Frame {frame_idx} not found, skipping...")
        return
    
    # Create images
    gray = cv2.imread(f"{frame_dir}/02_gray.png", 0)
    clahe = cv2.imread(f"{frame_dir}/03_clahe.png", 0)
    
    # Compute histograms
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe], [0], None, [256], [0, 256])
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: images
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Grayscale Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(clahe, cmap='gray')
    axes[0, 1].set_title('After CLAHE\n(clipLimit=2.0, grid=8x8)', fontweight='bold')
    axes[0, 1].axis('off')
    
    diff_clahe = cv2.absdiff(gray, clahe)
    axes[0, 2].imshow(diff_clahe, cmap='hot')
    axes[0, 2].set_title('Difference\n(enhancement)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Histogram
    axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 256), 
                    color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Original Histogram', fontweight='bold')
    axes[1, 0].set_xlabel('Gray Level')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(clahe.ravel(), bins=256, range=(0, 256), 
                    color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Histogram after CLAHE', fontweight='bold')
    axes[1, 1].set_xlabel('Gray level')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    
    # Overlap verification
    axes[1, 2].hist(gray.ravel(), bins=256, range=(0, 256), 
                    color='blue', alpha=0.5, label='Originale')
    axes[1, 2].hist(clahe.ravel(), bins=256, range=(0, 256), 
                    color='green', alpha=0.5, label='CLAHE')
    axes[1, 2].set_title('Histogram comparison', fontweight='bold')
    axes[1, 2].set_xlabel('Gray Level')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    # # Statistiche
    # mean_gray = np.mean(gray)
    # std_gray = np.std(gray)
    # mean_clahe = np.mean(clahe)
    # std_clahe = np.std(clahe)
    
    # stats_text = f"Statistiche:\n"
    # stats_text += f"Original - Mean: {mean_gray:.1f}, Std: {std_gray:.1f}\n"
    # stats_text += f"CLAHE - Mean: {mean_clahe:.1f}, Std: {std_clahe:.1f}"
    
    # fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # plt.suptitle('Pre-processing: CLAHE per Equalizzazione Adattiva del Contrasto', 
    #              fontsize=14, fontweight='bold')
    # plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/02_illumination_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 02_illumination_analysis.png")


def figure_03_edge_comparison(frame_idx: int):
    """Comparision edge detection operators."""
    print(f"[3/9] Generating comparison edge detection...")
    
    frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
    
    if not os.path.exists(frame_dir):
        print(f" Frame {frame_idx} not found, skipping...")
        return
    
    # upload frame blur
    gray_blur = cv2.imread(f"{frame_dir}/04_blur.png", 0)
    
    # Compute edges with different operators
    # Sobel
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobelx**2 + sobely**2)
    edges_sobel = np.clip(edges_sobel, 0, 255).astype(np.uint8)
    
    # Laplacian
    edges_laplacian = cv2.Laplacian(gray_blur, cv2.CV_16S, ksize=3)
    edges_laplacian = cv2.convertScaleAbs(edges_laplacian)
    
    # Prewitt
    kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 3.0
    kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 3.0
    prewittx = cv2.filter2D(gray_blur.astype(np.float32), -1, kernelx)
    prewitty = cv2.filter2D(gray_blur.astype(np.float32), -1, kernely)
    edges_prewitt = np.sqrt(prewittx**2 + prewitty**2)
    edges_prewitt = np.clip(edges_prewitt, 0, 255).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Edges
    axes[0, 0].imshow(edges_laplacian, cmap='gray')
    axes[0, 0].set_title('Laplacian\n(2° derivative, noise sensibility)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edges_prewitt, cmap='gray')
    axes[0, 1].set_title('Prewitt\n(1° derivative + smoothing)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges_sobel, cmap='gray')
    axes[0, 2].set_title('Sobel (Chosen)\n(1° derivative  + smoothing)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2:  Intensity Histogram
    axes[1, 0].hist(edges_laplacian.ravel(), bins=100, color='red', alpha=0.7)
    axes[1, 0].set_title('Laplacian Distribution')
    axes[1, 0].set_xlabel('Edge intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(edges_prewitt.ravel(), bins=100, color='orange', alpha=0.7)
    axes[1, 1].set_title('Prewitt Distribution')
    axes[1, 1].set_xlabel('Edge intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    
    axes[1, 2].hist(edges_sobel.ravel(), bins=100, color='green', alpha=0.7)
    axes[1, 2].set_title('Sobel Distribution')
    axes[1, 2].set_xlabel('Edge intensity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(alpha=0.3)
    
    # # Statistiche
    # stats_text = "Scelta: SOBEL perché combina smoothing e derivate,\n"
    # stats_text += "risultando più robusto al rumore rispetto a Laplacian"
    
    # fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
    #          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # plt.suptitle('Confronto Operatori di Edge Detection', fontsize=14, fontweight='bold')
    # plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/03_edge_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 03_edge_comparison.png")


def figure_04_morphology_steps(frame_idx: int, config: Dict):
    """Morphological operations sequence with correct parameters from config."""
    print(f"[4/9] Generating morphology analysis...")
    
    frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
    
    if not os.path.exists(frame_dir):
        print(f"  ⚠ Frame {frame_idx} not found, skipping...")
        return
    
    # Load masks
    fg_raw = cv2.imread(f"{frame_dir}/07_foreground_raw.png", 0)
    fg_person = cv2.imread(f"{frame_dir}/08_fg_person.png", 0)
    
    # Get CORRECT parameters from saved config
    min_area = config.get('min_area_person', 500)
    close_kernel_size = config.get('close_kernel_person', 9)
    open_kernel_size = config.get('open_kernel_person', 5)
    close_iter = config.get('close_iter_person', 5)
    open_iter = config.get('open_iter_person', 4)
    
    # Reconstruct intermediate steps with CORRECT parameters
    # Area opening
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg_raw, connectivity=8)
    fg_area = np.zeros_like(fg_raw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            fg_area[labels == i] = 255
    
    # Closing with CORRECT parameters
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    fg_closed = cv2.morphologyEx(fg_area, cv2.MORPH_CLOSE, k_close, iterations=close_iter)
    
    # Opening (final is fg_person, but we can also compute it)
    fg_opened = cv2.morphologyEx(fg_closed, cv2.MORPH_OPEN, k_open, iterations=open_iter)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    images = [
        (fg_raw, f'Raw Foreground\n(after change detection)'),
        (fg_area, f'After Area-Opening\n(min_area={min_area} pixels)'),
        (fg_closed, f'After Closing\n(ELLIPSE {close_kernel_size}×{close_kernel_size}, iter={close_iter})'),
        (fg_opened, f'After Opening\n(ELLIPSE {open_kernel_size}×{open_kernel_size}, iter={open_iter})'),
        (k_close * 255, f'Closing Kernel\n(ELLIPSE {close_kernel_size}×{close_kernel_size})'),
        (k_open * 255, f'Opening Kernel\n(ELLIPSE {open_kernel_size}×{open_kernel_size})'),
    ]
    
    for idx, (img, title) in enumerate(images):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')
    
    # Explanatory text
    text = f"Recommended sequence (from project requirements):\n"
    text += f"1. Area-Opening (λ={min_area}): removes small blobs (noise)\n"
    text += f"2. Closing ({close_kernel_size}×{close_kernel_size}, n={close_iter}): fills holes (body parts vs background)\n"
    text += f"3. Opening ({open_kernel_size}×{open_kernel_size}, n={open_iter}): final smoothing, removes protrusions"
    
    fig.text(0.5, 0.02, text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('Morphological Operations Sequence: Area-Opening → Closing → Opening', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/04_morphology_steps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 04_morphology_steps.png")


def figure_05_contour_refinement(frame_idx: int):
    """Visualization contour refinement."""
    print(f"[5/9] Generating analysis contour refinement...")
    
    frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
    
    if not os.path.exists(frame_dir):
        print(f" Frame {frame_idx} not found, skipping...")
        return
    
    # load frame e mask
    frame = cv2.imread(f"{frame_dir}/01_original.png")
    fg = cv2.imread(f"{frame_dir}/08_fg_person.png", 0)
    
    # found contourn grezzi
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("  No content find, skipping...")
        return
    
    # Take the biggest contour 
    cnt = max(contours, key=cv2.contourArea)
    
    # approxPolyDP
    epsilon = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Rough outline
    result1 = frame.copy()
    cv2.drawContours(result1, [cnt], -1, (0, 255, 0), 2)
    for pt in cnt[::5]:  # Show some point
        cv2.circle(result1, tuple(pt[0]), 2, (255, 0, 0), -1)
    
    axes[0].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Rough outline\n({len(cnt)} point, zigzag)', fontweight='bold')
    axes[0].axis('off')
    
    # 2. Dopo approxPolyDP
    result2 = frame.copy()
    cv2.drawContours(result2, [approx], -1, (0, 255, 0), 2)
    for pt in approx:
        cv2.circle(result2, tuple(pt[0]), 4, (255, 0, 0), -1)
    
    axes[1].imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'After approxPolyDP\n({len(approx)} points, epsilon=0.004)', fontweight='bold')
    axes[1].axis('off')
    
    # 3. Confronto sovrapposto
    result3 = frame.copy()
    cv2.drawContours(result3, [cnt], -1, (255, 0, 0), 1)  # Grezzo blu
    cv2.drawContours(result3, [approx], -1, (0, 255, 0), 2)  # Smoothed verde
    
    axes[2].imshow(cv2.cvtColor(result3, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlapped comparison\n(blu=Rough, green=smoothed)', fontweight='bold')
    axes[2].axis('off')
    
    # Info
    reduction = (1 - len(approx) / len(cnt)) * 100
    text = f"Point reduction: {reduction:.1f}% ({len(cnt)} → {len(approx)} point)\n"
    text += "Advantages: smoother contourn, less noise, keep general form"
    
    fig.text(0.5, 0.02, text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.suptitle('Contour Refinement with Polygonal approximation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(f'{OUTPUT_DIR}/05_contour_refinement.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 05_contour_refinement.png")


def figure_06_classification_features(features: List[Dict]):
    """Scatter plot features per classificazione."""
    print(f"[6/9] Generating analysis classificazione...")
    
    # Separa person e other
    person_data = [f for f in features if f['classification'] == 'person']
    other_data = [f for f in features if f['classification'] == 'other']
    
    if len(person_data) == 0 and len(other_data) == 0:
        print(" Nessuna feature trovata, skipping...")
        return
    
    # Estrai features
    p_area = [f['area'] for f in person_data]
    p_ar = [f['aspect_ratio'] for f in person_data]
    p_sol = [f['solidity'] for f in person_data]
    p_ext = [f['extent'] for f in person_data]
    
    o_area = [f['area'] for f in other_data]
    o_ar = [f['aspect_ratio'] for f in other_data]
    o_sol = [f['solidity'] for f in other_data]
    o_ext = [f['extent'] for f in other_data]
    
    # Plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Area vs Aspect Ratio
    ax1 = fig.add_subplot(gs[0, 0])
    if p_area:
        ax1.scatter(p_area, p_ar, c='green', s=50, alpha=0.6, label='Person')
    if o_area:
        ax1.scatter(o_area, o_ar, c='blue', s=50, alpha=0.6, label='Other')
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=1200, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=3000, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Auto threshold')
    ax1.set_xlabel('Area (pixel)', fontsize=11)
    ax1.set_ylabel('Aspect Ratio (H/W)', fontsize=11)
    ax1.set_title('Area vs Aspect Ratio\n(threshold: area≥1200, AR≥1.0; auto≥3000)', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Solidity vs Extent
    ax2 = fig.add_subplot(gs[0, 1])
    if p_sol:
        ax2.scatter(p_sol, p_ext, c='green', s=50, alpha=0.6, label='Person')
    if o_sol:
        ax2.scatter(o_sol, o_ext, c='blue', s=50, alpha=0.6, label='Other')
    ax2.axhline(y=0.10, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0.15, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Solidity', fontsize=11)
    ax2.set_ylabel('Extent', fontsize=11)
    ax2.set_title('Solidity vs Extent\n(threshold: sol≥0.15, ext≥0.10)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Area Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if p_area:
        ax3.hist(p_area, bins=20, color='green', alpha=0.6, label='Person')
    if o_area:
        ax3.hist(o_area, bins=20, color='blue', alpha=0.6, label='Other')
    ax3.axvline(x=1200, color='r', linestyle='--', linewidth=2, label='Min threshold')
    ax3.axvline(x=3000, color='orange', linestyle=':', linewidth=2, label='Auto threshold')
    ax3.set_xlabel('Area (pixel)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Area Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
#     # 4. Criteria (text box)
#     ax4 = fig.add_subplot(gs[1, 1])
#     ax4.axis('off')
    
#     criteria_text = f"""
# CLASSIFICATION CRITERIA

# PERSON ({len(person_data)} detections):

# AUTOMATIC (priority):
# ✓ Area ≥ 3000 pixels
# ✓ Height ≥ 80 pixels  
# ✓ Width ≥ 50 pixels
# ✓ Aspect Ratio ≥ 1.0
# → Classified as PERSON immediately

# STANDARD (if not automatic):
# ✓ Area ≥ 1200 pixels
# ✓ Height ≥ 80 pixels
# ✓ Aspect Ratio: 1.0 ≤ AR ≤ 4.0
# ✓ Solidity ≥ 0.15
# ✓ Extent ≥ 0.10
# ✓ H > W × 0.8

# OTHER ({len(other_data)} detections):
# • Fails PERSON criteria

# EXCLUSIONS (immediate rejection):
# ✗ W > H (wider than tall)
# ✗ Solidity < 0.12
# ✗ Extent < 0.08
# ✗ Area < 1200 or Height < 50
#     """
    
#     ax4.text(0.1, 0.9, criteria_text, fontsize=10, verticalalignment='top',
#              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Feature Analysis for Person/Other Classification', 
                 fontsize=14, fontweight='bold')
    plt.savefig(f'{OUTPUT_DIR}/06_classification_features.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 06_classification_features.png")


def figure_07_tracking_temporal(tracking: List[Dict]):
    """Visualizing temporal tracking."""
    print(f"[7/9] Generating visualization tracking...")
    
    # Filtra solo persone
    person_tracks = [t for t in tracking if t['classification'] == 'person']
    
    if len(person_tracks) == 0:
        print("No person tracking found, skipping...")
        return
    
    # Estrai traiettoria
    frames = [t['frame'] for t in person_tracks]
    cx = [t['cx'] for t in person_tracks]
    cy = [t['cy'] for t in person_tracks]
    
    # Carica background per plot
    bg = cv2.imread(f"{ANALYSIS_DIR}/background.png")
    # Check se è grayscale o già a colori
    if len(bg.shape) == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    else:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    
    # Plot
    fig = plt.figure(figsize=(15, 8))
    
    ax = plt.subplot(1, 1, 1)
    ax.imshow(bg, alpha=0.5)
    
    # Traiettoria
    ax.plot(cx, cy, 'r-', linewidth=2, label='Trajectory', alpha=0.7)
    
    # Punti colorati per tempo
    scatter = ax.scatter(cx[::5], cy[::5], c=frames[::5], cmap='plasma', s=50, 
                        edgecolors='white', linewidth=1, zorder=5)
    plt.colorbar(scatter, ax=ax, label='Frame')
    
    # Inizio e fine
    ax.scatter(cx[0], cy[0], c='green', s=200, marker='*', 
              edgecolors='white', linewidth=2, label='Start', zorder=10)
    ax.scatter(cx[-1], cy[-1], c='red', s=200, marker='X', 
              edgecolors='white', linewidth=2, label='End', zorder=10)
    
    ax.set_title(f'Person Trajectory during time({len(person_tracks)} frames)', 
                fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 320)
    ax.set_ylim(240, 0)
    
    plt.suptitle('Temporal Tracking with IoU Matching', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_tracking_temporal.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(" Saved: 07_tracking_temporal.png")


def figure_08_removed_objects():
    """Removed objects Analysis(Task 2)."""
    print(f"[8/9] Generating analysis removed objects...")
    
    # Cerca frame con removed objects
    # Per ora usiamo frame predefiniti
    frames_to_check = [380, 420]
    
    found = False
    for frame_idx in frames_to_check:
        frame_dir = f"{ANALYSIS_DIR}/frame_{frame_idx:04d}"
        if os.path.exists(frame_dir):
            found = True
            break
    
    if not found:
        print(" Frame per analysis removed objects not found,...")
        # Crea figura placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Task 2: Detection removal object\n\n'
                'Algorithm: Inverce difference (BG - Frame)\n'
                'When BG > Frame → Object present in BG but missing → REMOVED\n\n'
                'Execute the code in order to see real examples', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        ax.axis('off')
        plt.savefig(f'{OUTPUT_DIR}/08_removed_objects.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(" Saved: 08_removed_objects.png (placeholder)")
        return
    
    # Se trovato, carica immagini
    frame = cv2.imread(f"{frame_dir}/01_original.png")
    bg = cv2.imread(f"{ANALYSIS_DIR}/background.png", 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Frame {frame_idx}\n(dopo rimozione oggetto)', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(bg, cmap='gray')
    axes[1].set_title('Background\n(con oggetto originale)', fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle('Task 2: Detection Oggetti Rimossi (Differenza Inversa: BG - Frame)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_removed_objects.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 08_removed_objects.png")


def figure_09_summary_results(features: List[Dict], tracking: List[Dict]):
    """Result Summary."""
    print(f"[9/9] Generating results summary...")
    
    # Calcola statistiche
    total_frames = len(set([t['frame'] for t in tracking]))
    person_frames = len(set([t['frame'] for t in tracking if t['classification'] == 'person']))
    total_persons = len([f for f in features if f['classification'] == 'person'])
    total_objects = len([f for f in features if f['classification'] == 'other'])
    
    # Plot
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Statistiche
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    stats_text = f"""
    ═══════════════════════════════════════════════════════════════
                        STATISTICHE SISTEMA COMPLETO
    ═══════════════════════════════════════════════════════════════
    
    PROCESSED FRAMES:  {total_frames}
    
    PERSON:
      • Total Detection:     {total_persons}
      • Frames with person:   {person_frames} ({person_frames/total_frames*100:.1f}%)
    
    OBJECT:
      • Total Detection:     {total_objects}
      • Mean per frame:      {total_objects/total_frames:.2f}
    
    ALGORITHM:
      • Change Detection:     Background Subtraction + Edge (Sobel)
      • Morphology:          Area-Opening → Closing → Opening
      • Classification:      Geometric (area, AR, solidity)
      • Tracking:            IoU-based (threshold=0.2)
      • Task 2:              Removed object (diff inverce)
    
    CONFIGURATION:
      • Background:          Median {100} initial frame
      • Edge operator:       Sobel (3x3)
      • Connectivity:        8-neighborhood
      • Contour refinement:  approxPolyDP (epsilon=0.002)
    """
    
    ax1.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
             family='monospace', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 2. Detection nel tempo (histogram)
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Conta oggetti per frame
    frames_count = {}
    for t in tracking:
        frame = t['frame']
        if frame not in frames_count:
            frames_count[frame] = {'person': 0, 'other': 0}
        frames_count[frame][t['classification']] += 1
    
    frames = sorted(frames_count.keys())
    persons = [frames_count[f]['person'] for f in frames]
    others = [frames_count[f]['other'] for f in frames]
    
    ax2.bar(frames[::10], persons[::10], label='Person', color='green', alpha=0.6)
    ax2.bar(frames[::10], others[::10], bottom=persons[::10], 
           label='Object', color='blue', alpha=0.6)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Object Number')
    ax2.set_title('Temporal detection', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Pie chart classificazione
    ax3 = fig.add_subplot(gs[1, 1])
    labels = ['Person', 'Other']
    sizes = [total_persons, total_objects]
    colors = ['green', 'blue']
    explode = (0.1, 0)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax3.set_title('Distribution Classification', fontweight='bold')
    
    plt.suptitle('Summary: Complete result of the system', 
                 fontsize=16, fontweight='bold')
    plt.savefig(f'{OUTPUT_DIR}/09_summary_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 09_summary_results.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("REPORT GENERATOR - Generazione Figure da Dati Salvati")
    print("="*60)
    
    # Verifica che esistano i dati
    if not os.path.exists(ANALYSIS_DIR):
        print(f"\n ERRORE: Directory {ANALYSIS_DIR} non trovata!")
        print("\nDevi prima eseguire: python main_improved_with_analysis.py")
        print("Questo genererà i dati necessari per creare le figure.")
        return
    
    print(f"\n✓ Directory analysis trovata: {ANALYSIS_DIR}")
    print(f"Output figure: {OUTPUT_DIR}/\n")
    
    # Carica dati
    config, features, tracking, frame_data, background = load_analysis_data()
    
    # Frame da analizzare
    analysis_frames = config.get('analysis_frames', [120, 180, 250, 320, 380, 420])
    primary_frame = analysis_frames[2] if len(analysis_frames) > 2 else analysis_frames[0]
    
    print(f"\n Generazione figure in corso...\n")
    
    # Genera tutte le figure
    figure_01_pipeline_completa(380) # primary_frame
    figure_02_illumination_analysis(380, frame_data)
    figure_03_edge_comparison(380)
    figure_04_morphology_steps(380, config)  # ← Pass config
    figure_05_contour_refinement(380)
    figure_06_classification_features(features)
    figure_07_tracking_temporal(tracking)
    figure_08_removed_objects()
    figure_09_summary_results(features, tracking)
    
    print("\n" + "="*60)
    print("✓ COMPLETATO!")
    print(f"Tutte le figure sono state salvate in: {OUTPUT_DIR}/")
    print("\nFile generati:")
    for i in range(1, 10):
        filename = f"{i:02d}_*.png"
        print(f"  {filename}")
    print("\nOra puoi inserirle nella tua relazione!")
    print("="*60)


if __name__ == "__main__":
    main()