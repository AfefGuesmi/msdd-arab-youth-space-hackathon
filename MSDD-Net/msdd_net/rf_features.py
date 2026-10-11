# -*- coding: utf-8 -*-
"""
rf_features.py

Per-pixel feature extraction for the MARIDA Random-Forest baselines
(Kikaki et al. 2022). Faithful to the reference `engineering_patches.py`:

  RF_SS          : 11 Sentinel-2 bands (spectral signature)
  RF_SS+SI       : + 8 spectral indices in the ORIGINAL ORDER
                   NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI
  RF_SS+SI+GLCM  : + 6 GLCM texture features
                   CON, DIS, HOMO, ENER, COR, ASM
                   computed on the RGB composite (B4,B3,B2) converted to
                   grayscale via skimage.rgb2gray (0.2125, 0.7154, 0.0721),
                   clipped to [0, 0.15] and renormalized to [0, 1], then
                   quantized into 17 levels with np.digitize over
                   linspace(0, 1, 16). Window = 13x13, distance 1,
                   angles {0, 45, 90, 135} averaged.

The GLCM is computed densely for every pixel with integral-image box sums
over co-occurrence indicator images (one pass per angle). Mathematically
identical to running skimage.graycomatrix on each 13x13 window, but
~1000x faster.

Band order (MARIDA): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
"""
import numpy as np

BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# Original order used by engineering_patches.indices(): NDVI, FAI, FDI, SI,
# NDWI, NRD, NDMI, BSI. Order matters for downstream feature-importance
# plots that are compared against the paper.
SI_NAMES = ['NDVI', 'FAI', 'FDI', 'SI', 'NDWI', 'NRD', 'NDMI', 'BSI']

GLCM_NAMES = ['CON', 'DIS', 'HOMO', 'ENER', 'COR', 'ASM']

FEATURE_SETS = {
    'ss': BAND_NAMES,
    'ss_si': BAND_NAMES + SI_NAMES,
    'ss_si_glcm': BAND_NAMES + SI_NAMES + GLCM_NAMES,
}

# 0-indexed band positions within an (11, H, W) MARIDA stack.
_B2, _B3, _B4, _B6, _B8, _B11 = 1, 2, 3, 5, 7, 9

# Approximate Sentinel-2 central wavelengths (nm) used by the reference
# engineering_patches.py. Kept numeric-identical for reproducibility.
_L_NIR   = 833.0    # ~B8
_L_RED   = 665.0    # ~B4
_L_RE2   = 740.0    # ~B6 (red-edge)
_L_SWIR1 = 1614.0   # ~B11

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Spectral indices (match engineering_patches.py)
# ---------------------------------------------------------------------------
def spectral_indices(img):
    """
    img: (11, H, W) reflectance -> (8, H, W) float32, order = SI_NAMES.

    Exact formulas from engineering_patches.py:
      NDVI = (B8 - B4) / (B8 + B4)
      FAI  = B8 - [B4 + (B11 - B4) * (833 - 665) / (1614 - 665)]
      FDI  = B8 - [B6 + 10 * (B11 - B6) * (833 - 740) / (1614 - 740)]
      SI   = cbrt((1-B2)(1-B3)(1-B4))
      NDWI = (B3 - B8) / (B3 + B8)
      NRD  = B8 - B4
      NDMI = (B8 - B11) / (B8 + B11)
      BSI  = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    """
    b2, b3, b4 = img[_B2], img[_B3], img[_B4]
    b6, b8, b11 = img[_B6], img[_B8], img[_B11]

    ndvi = (b8 - b4) / (b8 + b4 + _EPS)

    # FAI baseline uses the RED wavelength (B4).
    r_acc_fai = b4 + (b11 - b4) * (_L_NIR - _L_RED) / (_L_SWIR1 - _L_RED)
    fai = b8 - r_acc_fai

    # FDI baseline uses the RED-EDGE wavelength (B6), not the red one.
    r_acc_fdi = b6 + 10.0 * (b11 - b6) * (_L_NIR - _L_RE2) / (_L_SWIR1 - _L_RE2)
    fdi = b8 - r_acc_fdi

    # Shadow Index. clip to [0, +inf) to guard against tiny negative FP noise
    # before the cube root (the reference assumes reflectance <= 1).
    si = np.cbrt(np.clip((1.0 - b2) * (1.0 - b3) * (1.0 - b4), 0.0, None))

    ndwi = (b3 - b8) / (b3 + b8 + _EPS)
    nrd = b8 - b4
    ndmi = (b8 - b11) / (b8 + b11 + _EPS)
    bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + _EPS)

    # ORDER matches SI_NAMES: NDVI, FAI, FDI, SI, NDWI, NRD, NDMI, BSI.
    return np.stack([ndvi, fai, fdi, si, ndwi, nrd, ndmi, bsi]).astype(np.float32)


# ---------------------------------------------------------------------------
# GLCM (dense, integral-image formulation; identical output to the
# reference loop that runs greycomatrix per 13x13 window)
# ---------------------------------------------------------------------------
def quantised_gray(img, lo=0.0, hi=0.15, num_bins=16):
    """
    Reproduce the reference preprocessing step exactly:
      1. Extract (B4, B3, B2) as an RGB composite.
      2. Clip to [lo, hi] and renormalize to [0, 1] by dividing by hi.
      3. Convert to grayscale using skimage.rgb2gray luminance weights
         (0.2125 R + 0.7154 G + 0.0721 B).
      4. Digitize into integer levels using np.digitize over
         np.linspace(0, 1, num_bins). Output range is [0, num_bins].

    The reference then sets GLCM `levels = num_bins + 1 = 17`.
    """
    rgb = np.stack([img[_B4], img[_B3], img[_B2]], axis=-1)
    rgb = np.clip(rgb, lo, hi) / hi  # -> [0, 1]

    # skimage.color.rgb2gray luminance weights (Rec. 709-ish).
    gray = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]

    # Matches np.digitize(gray, np.linspace(0, 1, num_bins)).
    bins = np.linspace(0.0, 1.0, num_bins)
    q = np.digitize(gray, bins)          # values in [0, num_bins]
    return q.astype(np.int64)


def _box_sum(a, y0, y1, x0, x1):
    """out[..., y, x] = sum a[..., y+y0 : y+y1+1, x+x0 : x+x1+1] (zero outside)."""
    C, H, W = a.shape
    p = max(abs(y0), abs(y1), abs(x0), abs(x1)) + 1
    ii = np.zeros((C, H + 2 * p + 1, W + 2 * p + 1), dtype=np.float64)
    ii[:, 1 + p:1 + p + H, 1 + p:1 + p + W] = a
    np.cumsum(ii, axis=1, out=ii)
    np.cumsum(ii, axis=2, out=ii)
    ya, yb = p + y0, p + y1 + 1
    xa, xb = p + x0, p + x1 + 1
    out = (ii[:, yb:yb + H, xb:xb + W] - ii[:, ya:ya + H, xb:xb + W]
           - ii[:, yb:yb + H, xa:xa + W] + ii[:, ya:ya + H, xa:xa + W])
    return out.astype(np.float32)


_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]  # 0, 45, 90, 135 deg, distance 1


def glcm_features(q, levels=17, window=13):
    """
    Dense per-pixel GLCM texture features.

    q: (H, W) integer image in [0, levels-1] (produced by quantised_gray
       with num_bins=16 -> range 0..16 -> levels=17).
    Returns (6, H, W) float32, order = GLCM_NAMES, averaged over 4 angles.
    """
    H, W = q.shape
    L = int(levels)
    i_idx, j_idx = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    i_f = i_idx.reshape(-1, 1, 1).astype(np.float32)
    j_f = j_idx.reshape(-1, 1, 1).astype(np.float32)
    diff = i_f - j_f
    out = np.zeros((6, H, W), dtype=np.float64)

    for dy, dx in _OFFSETS:
        # Pair code for every reference pixel whose neighbour is in-bounds.
        codes = np.full((H, W), -1, dtype=np.int64)
        ys = slice(max(0, -dy), H - max(0, dy))
        xs = slice(max(0, -dx), W - max(0, dx))
        yn = slice(max(0, -dy) + dy, H - max(0, dy) + dy)
        xn = slice(max(0, -dx) + dx, W - max(0, dx) + dx)
        codes[ys, xs] = q[ys, xs] * L + q[yn, xn]

        # One-hot stack (L*L, H, W), symmetric GLCM -> also add the transpose.
        onehot = np.zeros((L * L, H, W), dtype=np.float32)
        valid = codes >= 0
        yy, xx = np.nonzero(valid)
        c = codes[valid]
        onehot[c, yy, xx] += 1
        onehot[(c % L) * L + c // L, yy, xx] += 1

        # Sum only over reference pixels whose offset neighbour is also
        # inside the sliding window (matches skimage's per-window GLCM).
        h = window // 2
        y0, y1 = -h + max(0, -dy), h - max(0, dy)
        x0, x1 = -h + max(0, -dx), h - max(0, dx)
        counts = _box_sum(onehot, y0, y1, x0, x1)
        total = counts.sum(0, keepdims=True)
        P = counts / np.maximum(total, _EPS)

        con = (P * diff ** 2).sum(0)
        dis = (P * np.abs(diff)).sum(0)
        homo = (P / (1.0 + diff ** 2)).sum(0)
        asm = (P ** 2).sum(0)
        ener = np.sqrt(asm)
        mu_i = (P * i_f).sum(0)
        mu_j = (P * j_f).sum(0)
        sd_i = np.sqrt(np.maximum((P * (i_f - mu_i) ** 2).sum(0), 0))
        sd_j = np.sqrt(np.maximum((P * (j_f - mu_j) ** 2).sum(0), 0))
        cov = (P * (i_f - mu_i) * (j_f - mu_j)).sum(0)
        # skimage convention: correlation = 1 for a constant window.
        cor = np.where((sd_i < 1e-6) | (sd_j < 1e-6), 1.0, cov / np.maximum(sd_i * sd_j, _EPS))
        out += np.stack([con, dis, homo, ener, cor, asm])

    return (out / len(_OFFSETS)).astype(np.float32)


# ---------------------------------------------------------------------------
def extract_features(img, feature_set='ss_si_glcm', impute=None):
    """
    img: (11, H, W) reflectance (NaNs allowed; imputed with `impute`
    per-band values if given, else 0).
    Returns (F, H, W) float32 where F = len(FEATURE_SETS[feature_set]).
    """
    img = img.astype(np.float32, copy=True)
    nan = np.isnan(img)
    if nan.any():
        fill = np.zeros(img.shape[0], np.float32) if impute is None else np.asarray(impute, np.float32)
        img[nan] = np.broadcast_to(fill[:, None, None], img.shape)[nan]

    parts = [img]
    if feature_set in ('ss_si', 'ss_si_glcm'):
        parts.append(spectral_indices(img))
    if feature_set == 'ss_si_glcm':
        parts.append(glcm_features(quantised_gray(img)))
    feats = np.concatenate(parts, 0)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)