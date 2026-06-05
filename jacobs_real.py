import cv2
import math
import numpy as np
from statistics import mean, stdev


def draw_jacobs_grid(image_bgr, rows, cols, sampled_cells=None, excluded_cells=None):
    """Draws a numbered grid over image_bgr. Highlights sampled (green) and excluded (red) cells."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    sampled_cells = sampled_cells or set()
    excluded_cells = excluded_cells or set()

    overlay = img.copy()
    cell_num = 1
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols
            if cell_num in excluded_cells:
                cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (30, 30, 200), -1)
            elif cell_num in sampled_cells:
                cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (34, 240, 107), -1)
            cell_num += 1
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    for r in range(rows + 1):
        cv2.line(img, (0, r * h // rows), (w, r * h // rows), (200, 200, 200), 1, cv2.LINE_AA)
    for c in range(cols + 1):
        cv2.line(img, (c * w // cols, 0), (c * w // cols, h), (200, 200, 200), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.28, min(0.55, min(w // cols, h // rows) / 80))
    cell_num = 1
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h // rows, (r + 1) * h // rows
            x1, x2 = c * w // cols, (c + 1) * w // cols
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = str(cell_num)
            (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
            cv2.rectangle(img, (cx - tw // 2 - 2, cy - th - 2), (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
            cv2.putText(img, label, (cx - tw // 2, cy), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
            cell_num += 1

    return img


# t crítico bilateral 95% para df = 1..120 (tabela completa, sem lacunas)
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}


def _t_critical(df: int) -> float:
    """Returns 95% two-tailed t critical value for given degrees of freedom."""
    if df in _T_TABLE:
        return _T_TABLE[df]
    # linear interpolation between bracketing entries for gaps (e.g. df=35)
    keys = sorted(_T_TABLE)
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo < df < hi:
            frac = (df - lo) / (hi - lo)
            return _T_TABLE[lo] + frac * (_T_TABLE[hi] - _T_TABLE[lo])
    # beyond df=120: converges to z=1.960
    return 1.960


def draw_jacobs_heatmap(image_bgr, rows, cols, cell_counts, avg_per_cell, excluded_cells=None):
    """
    Density heatmap overlay. cell_counts: {cell_num: int} for counted cells.
    Counted cells are colored by count (blue→red). Unsampled crowd cells show
    the estimated avg at very low opacity. Excluded cells get a dark overlay.
    """
    excluded_cells = excluded_cells or set()
    img = image_bgr.copy()
    h, w = img.shape[:2]

    all_counts = [v for v in cell_counts.values() if v is not None]
    max_count = max(all_counts) if all_counts else 1

    def _heat_color(count):
        norm = min(count / max_count, 1.0) if max_count > 0 else 0.0
        # blue → cyan → green → yellow → red
        if norm < 0.25:
            t = norm / 0.25
            r, g, b = 0, int(t * 255), 255
        elif norm < 0.5:
            t = (norm - 0.25) / 0.25
            r, g, b = 0, 255, int((1 - t) * 255)
        elif norm < 0.75:
            t = (norm - 0.5) / 0.25
            r, g, b = int(t * 255), 255, 0
        else:
            t = (norm - 0.75) / 0.25
            r, g, b = 255, int((1 - t) * 255), 0
        return (b, g, r)  # BGR

    overlay = img.copy()
    cell_num = 1
    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * h // rows, (row + 1) * h // rows
            x1, x2 = col * w // cols, (col + 1) * w // cols
            if cell_num in excluded_cells:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
                alpha = 0.65
            elif cell_num in cell_counts:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), _heat_color(cell_counts[cell_num]), -1)
                alpha = 0.55
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), _heat_color(avg_per_cell), -1)
                alpha = 0.20
            roi = img[y1:y2, x1:x2]
            cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, roi, 1 - alpha, 0, roi)
            cell_num += 1

    for r in range(rows + 1):
        cv2.line(img, (0, r * h // rows), (w, r * h // rows), (160, 160, 160), 1, cv2.LINE_AA)
    for c in range(cols + 1):
        cv2.line(img, (c * w // cols, 0), (c * w // cols, h), (160, 160, 160), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.28, min(0.55, min(w // cols, h // rows) / 80))
    cell_num = 1
    for row in range(rows):
        for col in range(cols):
            if cell_num in cell_counts:
                y1, y2 = row * h // rows, (row + 1) * h // rows
                x1, x2 = col * w // cols, (col + 1) * w // cols
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = str(cell_counts[cell_num])
                (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
                cv2.rectangle(img, (cx - tw // 2 - 2, cy - th - 2), (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
                cv2.putText(img, label, (cx - tw // 2, cy), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
            cell_num += 1

    return img


def jacobs_estimate(sampled_counts, excluded_count, total_cells):
    """
    Crowd estimate via Herbert Jacobs grid method.

    sampled_counts : list[int]  people counted in each sampled cell
    excluded_count : int        cells excluded (stage, exits, empty areas)
    total_cells    : int        total grid cells

    Applies Finite Population Correction (FPC) so that margin → 0 as
    sample coverage → 100%. The reported interval assumes cells were chosen
    representatively; convenience sampling may introduce bias not captured here.
    """
    if not sampled_counts:
        return None

    crowd_cells = total_cells - excluded_count
    if crowd_cells <= 0:
        return None

    n = len(sampled_counts)
    avg = mean(sampled_counts)
    estimate = avg * crowd_cells

    if n > 1:
        s = stdev(sampled_counts)
        # FPC: sqrt((N-n)/(N-1)) — shrinks SE to 0 when n == crowd_cells
        fpc = math.sqrt((crowd_cells - n) / (crowd_cells - 1)) if crowd_cells > 1 else 0.0
        se = (s / math.sqrt(n)) * crowd_cells * fpc
        t = _t_critical(n - 1)
        margin = t * se
    else:
        s = 0.0
        # Single sample: no variance info — conservative 35% bound
        margin = estimate * 0.35

    return {
        "estimate": int(round(estimate)),
        "margin": int(round(margin)),
        "lower": max(0, int(round(estimate - margin))),
        "upper": int(round(estimate + margin)),
        "avg_per_cell": round(avg, 1),
        "std_per_cell": round(s, 1) if n > 1 else None,
        "sampled_cells": n,
        "crowd_cells": crowd_cells,
        "excluded_cells": excluded_count,
        "total_cells": total_cells,
        "coverage_pct": round(n / crowd_cells * 100, 1) if crowd_cells > 0 else 0.0,
    }
