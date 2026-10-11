r"""
generate_qualitative_results.py

Runs inference on the MADOS test set using a trained MSSD-Net checkpoint
and saves qualitative result triplets (input / label / prediction) as PNGs.

Reuses the exact same model-building, checkpoint-loading, and dataset logic
as evaluate_on_mados.py, so the images are guaranteed consistent with the
numbers in logs/*_results.json.

Usage (PowerShell):
    python generate_qualitative_results.py `
        --variant full `
        --model_path trained_models\swin-unet-mados\CE-Dice\swin_unet_full_pretrained.pth `
        --mados_path "C:\Users\afefg\OneDrive\Bureau\MSSD-Framework\MADOS_nearest" `
        --mados_splits_path "C:\Users\afefg\OneDrive\Bureau\MSSD-Framework\MADOS_nearest\splits" `
        --use_spectral_indices True `
        --use_texture_features True `
        --n_samples 20
"""

# --- Windows / Anaconda / PyTorch OpenMP fix -- MUST be before other imports ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "utils"))

from utils.config import Config
from mssd_net import build_mssd_net
from dataloader import (
    BANDS_MEAN, BANDS_STD, SPECTRAL_INDEX_NAMES, TEXTURE_FEATURE_NAMES,
)
from mados_dataloader import MADOSDataset, MARIDA_LABELS

# Same 6 property names as evaluate_on_mados.py
GLCM_FEATURE_NAMES = ["contrast", "dissimilarity", "homogeneity",
                       "energy", "correlation", "ASM"]


# ------------------------------------------------------------------
# Per-class colors (11 MADOS/MARIDA classes)
# ------------------------------------------------------------------
# Official MARIDA palette (utils/assets.py color_mapping). The previous
# palette gave NatM and Shallow Water near-identical purples.
from matplotlib.colors import to_rgb
CLASS_COLORS = np.array([to_rgb(c) for c in [
    'red', 'green', 'limegreen', 'brown', 'orange', 'silver', 'navy',
    'gold', 'purple', 'darkkhaki', 'darkturquoise']], dtype=np.float32)

IGNORE_COLOR = np.array([0.05, 0.05, 0.05], dtype=np.float32)  # for label == -1


# ------------------------------------------------------------------
# Model builder — identical to evaluate_on_mados.py
# ------------------------------------------------------------------
def build_model(options, device):
    config = Config()
    swin_cfg = getattr(getattr(config, "MODEL", None), "SWIN", None)
    model = build_mssd_net(
        options["variant"],
        img_size=config.DATA.IMG_SIZE,
        patch_size=getattr(swin_cfg, "PATCH_SIZE", 4),
        in_chans=options["input_channels"],
        num_classes=options["output_channels"],
        embed_dim=getattr(swin_cfg, "EMBED_DIM", 96),
        depths=getattr(swin_cfg, "DEPTHS", [2, 2, 2, 2]),
        depths_decoder=getattr(swin_cfg, "DEPTHS_DECODER", [1, 2, 2, 2]),
        num_heads=getattr(swin_cfg, "NUM_HEADS", [3, 6, 12, 24]),
        window_size=getattr(swin_cfg, "WINDOW_SIZE", 8),
    )
    model.to(device)
    return model


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def un_normalize_rgb(tensor_chw, rgb_indices=(3, 2, 1)):
    """
    Extract RGB channels from a normalized input tensor.
    Sentinel-2 11-band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    so B4 (Red) = idx 3, B3 (Green) = idx 2, B2 (Blue) = idx 1.
    """
    rgb = tensor_chw[list(rgb_indices)]
    mean = np.array([BANDS_MEAN[i] for i in rgb_indices], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([BANDS_STD[i]  for i in rgb_indices], dtype=np.float32).reshape(3, 1, 1)
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def stretch_channel(arr, lo_pct=2, hi_pct=98):
    a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def colorize(label_map, base=None):
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:] = IGNORE_COLOR if base is None else base
    for c in range(len(CLASS_COLORS)):
        rgb[label_map == c] = CLASS_COLORS[c]
    return rgb


def save_legend(out_path):
    fig, ax = plt.subplots(figsize=(4.5, 6))
    for i, (name, col) in enumerate(zip(MARIDA_LABELS, CLASS_COLORS)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=col, edgecolor="black"))
        ax.text(1.15, i + 0.5, f"{i}. {name}", va="center", fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, len(CLASS_COLORS))
    ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_triplet(scene_dir, rgb, label, pred, scene_name):
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Labels are sparse: draw them over a dimmed RGB so unlabelled
    # pixels stay readable instead of an all-black panel.
    label_rgb = colorize(label, base=rgb * 0.6)
    pred_rgb  = colorize(pred)

    plt.imsave(scene_dir / "input.png",      rgb)
    plt.imsave(scene_dir / "label.png",      label_rgb)
    plt.imsave(scene_dir / "prediction.png", pred_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb);       axes[0].set_title(f"Input — {scene_name}")
    axes[1].imshow(label_rgb); axes[1].set_title("Reference label")
    axes[2].imshow(pred_rgb);  axes[2].set_title(f"MSSD-Net prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(scene_dir / "combined.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(options):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    effective_use_texture = options["use_texture_features"] and not options["use_glcm_texture"]
    n_extra = (
        (len(SPECTRAL_INDEX_NAMES) if options["use_spectral_indices"] else 0)
        + (len(GLCM_FEATURE_NAMES) if options["use_glcm_texture"] else
           (len(TEXTURE_FEATURE_NAMES) if effective_use_texture else 0))
    )
    options["input_channels"] = 11 + n_extra
    print(f"Input channels = {options['input_channels']} (base 11 + extras {n_extra})")

    model = build_model(options, device)
    print(f"Loading checkpoint: {options['model_path']}")
    checkpoint = torch.load(options["model_path"], map_location=device)
    state_dict = checkpoint["model_state_dict"] if (
        isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    ) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded OK.")

    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)
    transform_det = transforms.Compose([transforms.ToTensor()])

    dataset = MADOSDataset(
        options["mados_path"],
        split=options["mados_split"],
        transform=transform_det,
        use_spectral_indices=options["use_spectral_indices"],
        use_texture_features=options["use_texture_features"],
        use_glcm_texture=options["use_glcm_texture"],
        standardization=standardization,
        splits_path=options["mados_splits_path"],
    )
    print(f"Loaded {len(dataset)} MADOS patches (split={options['mados_split']})")

    loader = DataLoader(dataset, batch_size=options["batch"], shuffle=False)

    out_root = Path(options["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    save_legend(out_root / "legend.png")

    saved = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="inference")):
            batch_start = batch_idx * options["batch"]
            images_dev = images.to(device)
            probs = torch.nn.functional.softmax(model(images_dev), dim=1)
            preds = probs.argmax(1).cpu().numpy()
            images_cpu = images.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for b in range(images.shape[0]):
                if saved >= options["n_samples"]:
                    break

                scene_name = f"patch_{saved:04d}"
                if hasattr(dataset, "rois"):
                    scene_name += "_" + str(dataset.rois[batch_start + b])
                scene_dir = out_root / scene_name

                rgb = un_normalize_rgb(images_cpu[b])
                rgb = np.stack([stretch_channel(rgb[..., i]) for i in range(3)], axis=-1)

                label = targets_np[b].astype(np.int64)
                pred  = preds[b].astype(np.int64)
                # Skip patches with (almost) no reference labels, or no
                # debris when --require_debris is set: an empty reference
                # panel says nothing about model quality.
                if (label >= 0).sum() < options["min_labeled_px"]:
                    continue
                if options["require_debris"] and not (label == 0).any():
                    continue

                save_triplet(scene_dir, rgb, label, pred, scene_name)
                saved += 1

            if saved >= options["n_samples"]:
                break

    print(f"\nDone. {saved} triplets saved to: {out_root.resolve()}")
    print("Open the folder, review the PNGs, then upload to Google Drive.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "full", "only_residual"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mados_path", required=True)
    parser.add_argument("--mados_splits_path", required=True)
    parser.add_argument("--mados_split", default="test")
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--output_channels", default=11, type=int)
    parser.add_argument("--use_spectral_indices", default=False, type=ast.literal_eval)
    parser.add_argument("--use_texture_features", default=False, type=ast.literal_eval)
    parser.add_argument("--use_glcm_texture", default=False, type=ast.literal_eval)
    parser.add_argument("--n_samples", default=20, type=int)
    parser.add_argument("--min_labeled_px", default=50, type=int,
                        help="skip patches with fewer labelled pixels than this")
    parser.add_argument("--require_debris", default=True, type=ast.literal_eval,
                        help="only keep patches containing labelled Marine Debris")
    parser.add_argument("--out_dir", default="qualitative_results")
    args = parser.parse_args()
    main(vars(args))