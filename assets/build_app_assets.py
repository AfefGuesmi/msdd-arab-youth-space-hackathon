r"""
build_app_assets.py

For each checkpoint variant (baseline, full, only_residual):
  - pick the TOP N patches with the most debris pixels (best detections)
  - copy only those into assets/qualitative/<variant>/

Then generate assets/debris_reports.csv with the fields the app needs:
  Debris Type, Danger Level, Geolocation, Recommended Action, Report Date.

Run from the repo root:
    python build_app_assets.py
"""
from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
REPO_ROOT      = Path(r"C:\Users\afefg\OneDrive\Bureau\msdd-arab-youth-space-hackathon")
SOURCE_RESULTS = Path(r"C:\Users\afefg\OneDrive\Bureau\MSSD-Framework\qualitative_results")

VARIANTS = ["baseline", "full", "only_residual"]
TOP_N = 5  # how many best patches to keep per variant

DEST_QUAL = REPO_ROOT / "assets" / "qualitative"
DEST_CSV  = REPO_ROOT / "assets" / "debris_reports.csv"

# Color of "Marine Debris" (class 0) in prediction.png
# Matches CLASS_COLORS[0] = [230, 25, 75] / 255.0
DEBRIS_RGB = np.array([230, 25, 75], dtype=np.float32) / 255.0
COLOR_TOL  = 0.06

# Approximate AOI center — replace with real coords if you have the MADOS scene map
AOI_CENTER_LAT = 24.10
AOI_CENTER_LON = 53.77


# ============================================================
# SCORE A PATCH — count debris pixels
# ============================================================
def debris_score(pred_png: Path) -> tuple[int, int]:
    """Return (debris_pixels, total_pixels) for a prediction.png."""
    img = np.array(Image.open(pred_png).convert("RGB"), dtype=np.float32) / 255.0
    dist = np.linalg.norm(img - DEBRIS_RGB, axis=-1)
    mask = dist < COLOR_TOL
    return int(mask.sum()), int(img.shape[0] * img.shape[1])


# ============================================================
# DANGER LEVEL RULES
# ============================================================
def classify(pct_debris: float) -> tuple[str, str]:
    """Return (danger_level, recommended_action)."""
    if pct_debris >= 5.0:
        return "High",   "Deploy specialized cleanup vessels"
    if pct_debris >= 1.0:
        return "Medium", "Schedule inspection survey"
    return "Low",    "Monitor via routine satellite pass"


# ============================================================
# STEP 1 — Find top-N patches per variant
# ============================================================
def select_top_patches():
    selections = {}   # variant -> list of (score, total, patch_dir)

    for variant in VARIANTS:
        src = SOURCE_RESULTS / variant
        if not src.exists():
            print(f"  ! Variant folder missing: {src} — skipping")
            continue

        patch_dirs = sorted([p for p in src.iterdir() if p.is_dir() and p.name.startswith("patch_")])
        if not patch_dirs:
            print(f"  ! No patch_* in {src} — skipping")
            continue

        scored = []
        for pd in patch_dirs:
            pred = pd / "prediction.png"
            if not pred.exists():
                continue
            n_debris, n_total = debris_score(pred)
            scored.append((n_debris, n_total, pd))

        scored.sort(key=lambda x: x[0], reverse=True)   # most debris first
        top = scored[:TOP_N]
        selections[variant] = top
        print(f"  [OK] {variant}: selected top {len(top)} of {len(scored)} patches")
        for n, t, pd in top:
            print(f"       {pd.name}  ->  {n} debris px  ({n/t*100:.2f} %)")

    return selections


# ============================================================
# STEP 2 — Copy selected patches into assets/qualitative/<variant>/
# ============================================================
def copy_selection(selections):
    if DEST_QUAL.exists():
        shutil.rmtree(DEST_QUAL)
    DEST_QUAL.mkdir(parents=True, exist_ok=True)

    for variant, top in selections.items():
        # Copy legend
        legend_src = SOURCE_RESULTS / variant / "legend.png"
        if legend_src.exists():
            shutil.copy2(legend_src, DEST_QUAL / f"legend_{variant}.png")

        # Copy each selected patch
        var_dir = DEST_QUAL / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        for n, t, pd in top:
            target = var_dir / pd.name
            shutil.copytree(pd, target)

    print(f"[OK] Copied selected patches to {DEST_QUAL}")


# ============================================================
# STEP 3 — Build CSV
# ============================================================
def build_csv(selections):
    rows = []
    row_idx = 0

    for variant, top in selections.items():
        for n_debris, n_total, pd in top:
            pct = (n_debris / n_total) * 100.0
            danger, action = classify(pct)

            # Approximate geolocation (grid spread around AOI center)
            lat = AOI_CENTER_LAT + (row_idx % 5) * 0.01
            lon = AOI_CENTER_LON + (row_idx // 5) * 0.01

            rows.append({
                "variant":            variant,
                "patch_id":           pd.name,
                "debris_type":        "Plastic",
                "danger_level":       danger,
                "debris_pixels":      n_debris,
                "debris_percent":     f"{pct:.2f}",
                "latitude":           f"{lat:.4f}",
                "longitude":          f"{lon:.4f}",
                "recommended_action": action,
                "report_date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            row_idx += 1

    DEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "patch_id", "debris_type", "danger_level",
                  "debris_pixels", "debris_percent",
                  "latitude", "longitude",
                  "recommended_action", "report_date"]
    with open(DEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {DEST_CSV}")

    highs = sum(1 for r in rows if r["danger_level"] == "High")
    meds  = sum(1 for r in rows if r["danger_level"] == "Medium")
    lows  = sum(1 for r in rows if r["danger_level"] == "Low")
    print(f"     Danger breakdown: High={highs}, Medium={meds}, Low={lows}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Repo root : {REPO_ROOT}")
    print(f"Source    : {SOURCE_RESULTS}")
    print(f"Variants  : {VARIANTS}")
    print(f"Top N     : {TOP_N} per variant")
    print()
    print("Scanning patches for debris coverage...")
    selections = select_top_patches()
    print()
    print("Copying selected patches...")
    copy_selection(selections)
    print()
    print("Building CSV...")
    build_csv(selections)
    print()
    print("Done. Now run:")
    print("  git add assets/ build_app_assets.py")
    print("  git commit -m 'Add best-variant qualitative results and debris report CSV'")
    print("  git push origin msdd-net")