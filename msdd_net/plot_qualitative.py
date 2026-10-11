# -*- coding: utf-8 -*-
"""
plot_qualitative.py

Side-by-side qualitative comparison on the MARIDA test split:

    RGB | reference labels (over RGB) | model 1 | model 2 | ...

Fixes the problems of the earlier qualitative figures:
  * patches are chosen by how much *labelled Marine Debris* they contain
    (the old script took the first N patches in file order, so many
    reference panels were completely empty / black);
  * MARIDA labels are sparse -> the reference panel is drawn as a coloured
    overlay on the RGB image, unlabelled pixels stay visible as imagery;
  * the official MARIDA class colours are used (red = debris), so no two
    classes share near-identical colours;
  * every model panel is annotated with Marine-Debris F1 and accuracy on
    that patch's labelled pixels, so the figure is self-explanatory;
  * a second "debris focus" figure shows only debris: labelled debris
    outlined in yellow, predicted debris filled in red, per model.

Model specs (any mix, any order):  NAME=PATH
  * PATH is a directory  -> Random-Forest predictions written by
                            train_rf.py (<dir>/<roi>.npy)
  * PATH is a .pth file  -> a checkpoint from train_marida.py (architecture
                            and input features are read from the stored
                            options). For older checkpoints without stored
                            options use NAME=mssd:<variant>:PATH or
                            NAME=unet:PATH.

Example:
  python plot_qualitative.py --data_path data \
      --models "U-Net=trained_models/unet/best_model.pth" \
               "RF SS+SI+GLCM=trained_models/rf_ss_si_glcm/predictions" \
               "MSSD-Net=trained_models/baseline/best_model.pth" \
      --n_patches 6 --out_dir ../qualitative_results/marida_comparison
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dataloader import (BANDS_MEAN, BANDS_STD, compute_spectral_indices,  # noqa: E402
                        compute_texture_features)
from train_rf import MARIDA_11, load_patch, read_split  # noqa: E402

# Official MARIDA colours (utils/assets.py color_mapping)
CLASS_COLORS = np.array([to_rgb(c) for c in [
    'red', 'green', 'limegreen', 'brown', 'orange', 'silver', 'navy',
    'gold', 'purple', 'darkkhaki', 'darkturquoise']], dtype=np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RFPredictions:
    kind = 'rf'

    def __init__(self, pred_dir):
        self.pred_dir = pred_dir

    def predict(self, roi, img):
        p = os.path.join(self.pred_dir, f'{roi}.npy')
        if not os.path.exists(p):
            raise FileNotFoundError(f'No RF prediction for {roi} in {self.pred_dir} '
                                    f'(run train_rf.py --predict_patches True)')
        return np.load(p).astype(np.int64)


class TorchModel:
    kind = 'dl'

    def __init__(self, spec, device):
        arch, variant, path = spec
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        opt = ckpt.get('options', {}) if isinstance(ckpt, dict) else {}
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        arch = arch or opt.get('model', 'mssd')
        variant = variant or opt.get('variant', 'baseline')
        self.si = bool(opt.get('use_spectral_indices', False))
        self.tex = bool(opt.get('use_texture_features', False))
        in_ch = 11 + 6 * self.si + 2 * self.tex
        n_cls = opt.get('num_classes', 11)
        if arch == 'unet':
            from unet import UNet
            self.model = UNet(in_ch, n_cls, opt.get('hidden_channels', 16))
        else:
            from mssd_net import build_mssd_net
            from utils.config import Config
            s = Config().MODEL.SWIN
            self.model = build_mssd_net(variant, img_size=256, patch_size=s.PATCH_SIZE, in_chans=in_ch,
                                        num_classes=n_cls, embed_dim=s.EMBED_DIM, depths=s.DEPTHS,
                                        num_heads=s.NUM_HEADS, window_size=s.WINDOW_SIZE)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.device = device

    def _prep(self, img):
        x = np.moveaxis(img, 0, -1).copy()                       # (H,W,11)
        nan = np.isnan(x)
        x[nan] = np.broadcast_to(BANDS_MEAN, x.shape)[nan]
        raw = (x - BANDS_MEAN) / BANDS_STD
        parts = [raw]
        if self.si:
            parts.append(compute_spectral_indices(x))
        if self.tex:
            parts.append(compute_texture_features(x))
        x = np.concatenate(parts, -1).astype(np.float32)
        return torch.from_numpy(np.moveaxis(x, -1, 0))[None].to(self.device)

    @torch.no_grad()
    def predict(self, roi, img):
        return self.model(self._prep(img)).argmax(1)[0].cpu().numpy()


def parse_model_spec(spec, device):
    name, _, rest = spec.partition('=')
    if not rest:
        raise ValueError(f'Bad --models entry {spec!r}; expected NAME=PATH')
    if os.path.isdir(rest):
        return name, RFPredictions(rest)
    bits = rest.split(':')
    if bits[0] == 'unet' and len(bits) == 2:
        return name, TorchModel(('unet', None, bits[1]), device)
    if bits[0] == 'mssd' and len(bits) == 3:
        return name, TorchModel(('mssd', bits[1], bits[2]), device)
    return name, TorchModel((None, None, rest), device)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def rgb_composite(img, lo=2, hi=98):
    rgb = np.nan_to_num(np.stack([img[3], img[2], img[1]], -1))
    out = np.empty_like(rgb)
    for c in range(3):
        a, b = np.percentile(rgb[..., c], [lo, hi])
        out[..., c] = np.clip((rgb[..., c] - a) / max(b - a, 1e-6), 0, 1)
    return out


def colorize(labels, base=None, alpha=1.0):
    out = np.zeros(labels.shape + (3,), np.float32) if base is None else base.copy()
    valid = labels >= 0
    col = CLASS_COLORS[np.clip(labels, 0, len(CLASS_COLORS) - 1)]
    out[valid] = (1 - alpha) * out[valid] + alpha * col[valid]
    return out


def debris_focus(rgb, gt, pred):
    out = rgb * 0.55
    out[pred == 0] = [1.0, 0.1, 0.1]
    gt_md = gt == 0
    edge = gt_md & ~ndimage.binary_erosion(gt_md)
    out[edge] = [1.0, 0.95, 0.0]
    return out


def patch_scores(gt, pred):
    valid = gt >= 0
    acc = (gt[valid] == pred[valid]).mean() if valid.any() else float('nan')
    t, p = gt[valid] == 0, pred[valid] == 0
    tp = (t & p).sum()
    f1 = 2 * tp / max(t.sum() + p.sum(), 1) if t.any() else float('nan')
    return f1, acc


def select_rois(data_path, splits_dir, n, explicit=None, min_gap_tiles=True):
    if explicit:
        return explicit
    rois = read_split(data_path, 'test', splits_dir)
    scored = []
    for r in rois:
        _, m, _ = load_patch(data_path, r)
        n_md, n_lab = int((m == 0).sum()), int((m >= 0).sum())
        n_cls = len(np.unique(m[m >= 0]))
        if n_md:
            scored.append((n_md + 5 * n_cls, r))
    scored.sort(reverse=True)
    chosen, seen_scenes = [], set()
    for _, r in scored:                    # prefer one patch per scene for diversity
        scene = r.rsplit('_', 1)[0]
        if min_gap_tiles and scene in seen_scenes:
            continue
        chosen.append(r)
        seen_scenes.add(scene)
        if len(chosen) == n:
            break
    for _, r in scored:                    # top up if not enough distinct scenes
        if len(chosen) == n:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen


def legend_handles():
    return [Patch(facecolor=CLASS_COLORS[i], edgecolor='k', label=MARIDA_11[i]) for i in range(11)]


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [parse_model_spec(s, device) for s in opt['models']]
    rois = select_rois(opt['data_path'], opt['splits_dir'], opt['n_patches'], opt['rois'])
    os.makedirs(opt['out_dir'], exist_ok=True)
    print('Patches:', rois)

    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(rois), ncol, figsize=(2.9 * ncol, 2.9 * len(rois) + 0.9), squeeze=False)
    fig2, axes2 = plt.subplots(len(rois), 1 + len(models), figsize=(2.9 * (1 + len(models)),
                               2.9 * len(rois) + 0.6), squeeze=False)
    summary = {}

    for r, roi in enumerate(rois):
        img, gt, _ = load_patch(opt['data_path'], roi)
        rgb = rgb_composite(img)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(roi, fontsize=8)
        axes[r, 1].imshow(colorize(gt, base=rgb * 0.6))
        axes2[r, 0].imshow(debris_focus(rgb, gt, np.full_like(gt, -1)))
        axes2[r, 0].set_ylabel(roi, fontsize=8)
        summary[roi] = {}
        for c, (name, model) in enumerate(models):
            pred = model.predict(roi, img)
            f1, acc = patch_scores(gt, pred)
            summary[roi][name] = {'md_f1': None if np.isnan(f1) else float(f1), 'acc': float(acc)}
            axes[r, 2 + c].imshow(colorize(pred))
            axes[r, 2 + c].set_xlabel(f'MD F1 {f1:.2f} | acc {acc:.2f}', fontsize=8)
            axes2[r, 1 + c].imshow(debris_focus(rgb, gt, pred))
            axes2[r, 1 + c].set_xlabel(f'MD F1 {f1:.2f}', fontsize=8)
            if r == 0:
                axes[0, 2 + c].set_title(name, fontsize=10)
                axes2[0, 1 + c].set_title(name, fontsize=10)

    axes[0, 0].set_title('RGB (B4-B3-B2)', fontsize=10)
    axes[0, 1].set_title('Reference labels', fontsize=10)
    axes2[0, 0].set_title('RGB + labelled debris', fontsize=10)
    for ax in list(axes.flat) + list(axes2.flat):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(handles=legend_handles(), loc='lower center', ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(opt['out_dir'], 'comparison_all_classes.png'), dpi=opt['dpi'])
    fig2.legend(handles=[Patch(facecolor='red', label='Predicted Marine Debris'),
                         Patch(facecolor='yellow', label='Labelled Marine Debris (outline)')],
                loc='lower center', ncol=2, fontsize=9, frameon=False)
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    fig2.savefig(os.path.join(opt['out_dir'], 'comparison_debris_focus.png'), dpi=opt['dpi'])
    with open(os.path.join(opt['out_dir'], 'patch_scores.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved figures to', os.path.abspath(opt['out_dir']))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_path', default=os.path.join(PROJECT_ROOT, 'data'))
    p.add_argument('--splits_dir', default='splits')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--n_patches', type=int, default=6)
    p.add_argument('--rois', nargs='*', default=None, help='explicit test ROIs, e.g. 1-12-19_48MYU_0')
    p.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, '..', 'qualitative_results', 'marida_comparison'))
    p.add_argument('--dpi', type=int, default=150)
    return vars(p.parse_args(argv))


if __name__ == '__main__':
    main(parse_args())
