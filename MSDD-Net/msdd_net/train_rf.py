# -*- coding: utf-8 -*-
"""
train_rf.py

Random-Forest baselines from the MARIDA paper (Kikaki et al. 2022).
Faithful reproduction of the reference `random_forest.py` + `train_eval.py`:

    --features ss           RF_SS           (11 bands)
    --features ss_si        RF_SS+SI        (+ 8 spectral indices)
    --features ss_si_glcm   RF_SS+SI+GLCM   (+ 6 GLCM textures)  <- best in paper

Paper setup reproduced EXACTLY:
  * Pipeline([StandardScaler, RandomForestClassifier])
  * RF: n_estimators=125, criterion='gini', max_depth=20,
        min_samples_leaf=1, min_impurity_decrease=0, oob_score=True,
        class_weight='balanced_subsample', random_state=5, n_jobs=-1
  * 11 aggregated classes (Waves, Cloud Shadows, Wakes, Mixed Water
    -> Marine Water)
  * Sample weights = 1/confidence-code (High=1.0, Moderate=0.5, Low=1/3)
  * Trained on annotated train pixels; evaluated on annotated val/test
    pixels.

Outputs (under --out_dir, default trained_models/rf_<features>/):
  rf.joblib                 the fitted pipeline
  results.json              per-class IoU / PA / F1 / precision + averages
  predictions/<roi>.npy     full-patch predictions (0..10, uint8) for the
                            test split, used by plot_qualitative.py

Pixel features are cached per split under --cache_dir so re-runs
(e.g. grid search) skip the GLCM extraction.

Example:
  python train_rf.py --data_path data --features ss_si_glcm --n_jobs 8
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import rasterio
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from rf_features import FEATURE_SETS, extract_features  # noqa: E402
from utils.cli import str2bool  # noqa: E402

MARIDA_11 = ['Marine Debris', 'Dense Sargassum', 'Sparse Sargassum', 'Natural Organic Material',
             'Ship', 'Clouds', 'Marine Water', 'Sediment-Laden Water', 'Foam', 'Turbid Water',
             'Shallow Water']
SHORT_11 = ['MD', 'DenS', 'SpS', 'NatM', 'Ship', 'Clouds', 'MWater', 'SLWater', 'Foam',
            'TWater', 'SWater']

# Reference conf_mapping = {'High': 1, 'Moderate': 2, 'Low': 3} with
# sample_weight = 1 / conf_mapping[conf].
CONF_WEIGHT = {1: 1.0, 2: 0.5, 3: 1.0 / 3.0}

BANDS_MEAN = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
                       0.03875053, 0.03436435, 0.0392113, 0.02358126, 0.01588816], np.float32)


# ---------------------------------------------------------------------------
# MARIDA I/O
# ---------------------------------------------------------------------------
def roi_paths(data_path, roi):
    folder = '_'.join(['S2'] + roi.split('_')[:-1])
    name = '_'.join(['S2'] + roi.split('_'))
    base = os.path.join(data_path, 'patches', folder, name)
    return base + '.tif', base + '_cl.tif', base + '_conf.tif'


def read_split(data_path, split, splits_dir='splits'):
    return list(np.genfromtxt(os.path.join(data_path, splits_dir, f'{split}_X.txt'),
                              dtype='str', ndmin=1))


def load_patch(data_path, roi):
    """
    Returns img (11,H,W) float32, mask (H,W) int64 in 0..10 or -1,
    conf (H,W) int64 (1=High, 2=Moderate, 3=Low; 1 if file missing).
    """
    img_p, cl_p, conf_p = roi_paths(data_path, roi)
    with rasterio.open(img_p) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(cl_p) as src:
        mask = src.read(1).astype(np.int64)
    conf = np.ones_like(mask)
    if os.path.exists(conf_p):
        with rasterio.open(conf_p) as src:
            conf = src.read(1).astype(np.int64)
    mask[np.isin(mask, [12, 13, 14, 15])] = 7   # water super-class
    mask = mask - 1                              # 0 (unlabelled) -> -1
    return img, mask, conf


def _patch_pixels(data_path, roi, feature_set):
    img, mask, conf = load_patch(data_path, roi)
    valid = mask >= 0
    if not valid.any():
        return None
    f = extract_features(img, feature_set, impute=BANDS_MEAN)
    return f[:, valid].T, mask[valid], conf[valid]


def build_pixel_table(data_path, split, feature_set, cache_dir, n_jobs, splits_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f'{splits_dir}_{split}_{feature_set}.npz')
    if os.path.exists(cache):
        d = np.load(cache)
        return d['X'], d['y'], d['conf']
    rois = read_split(data_path, split, splits_dir)
    t0 = time.time()
    parts = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_patch_pixels)(data_path, r, feature_set) for r in rois)
    parts = [p for p in parts if p is not None]
    X = np.concatenate([p[0] for p in parts]).astype(np.float32)
    y = np.concatenate([p[1] for p in parts])
    conf = np.concatenate([p[2] for p in parts])
    print(f'[{split}] {len(rois)} patches -> {len(y):,} labelled px, {X.shape[1]} features '
          f'({time.time() - t0:.0f}s)')
    np.savez_compressed(cache, X=X, y=y, conf=conf)
    return X, y, conf


# ---------------------------------------------------------------------------
# Metrics (paper Table 4: IoU, PA (= per-class recall), F1)
# ---------------------------------------------------------------------------
def per_class_scores(y_true, y_pred, n=11):
    cm = np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.nan_to_num(tp / (tp + fp + fn))
        pa = np.nan_to_num(tp / (tp + fn))
        prec = np.nan_to_num(tp / (tp + fp))
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    present = cm.sum(1) > 0
    res = {SHORT_11[i]: dict(iou=iou[i], pa=pa[i], precision=prec[i], f1=f1[i],
                             support=int(cm[i].sum()))
           for i in range(n)}
    res['Average'] = dict(iou=iou[present].mean(), pa=pa[present].mean(),
                          precision=prec[present].mean(), f1=f1[present].mean(),
                          support=int(cm.sum()))
    res['pixel_accuracy'] = float(tp.sum() / cm.sum())
    res['confusion_matrix'] = cm.tolist()
    return {k: ({kk: (int(vv) if kk == 'support' else float(vv)) for kk, vv in v.items()}
                if isinstance(v, dict) else v) for k, v in res.items()}


def format_table(res, title):
    lines = [title, f"{'Class':8s} {'IoU':>6s} {'PA':>6s} {'F1':>6s} {'Prec':>6s} {'Support':>9s}"]
    for k in SHORT_11 + ['Average']:
        m = res[k]
        lines.append(f"{k:8s} {m['iou']:6.2f} {m['pa']:6.2f} {m['f1']:6.2f} "
                     f"{m['precision']:6.2f} {m['support']:9d}")
    lines.append(f"Overall pixel accuracy: {res['pixel_accuracy']:.4f}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Model construction -- matches random_forest.py exactly
# ---------------------------------------------------------------------------
def build_rf(n_estimators, max_depth, n_jobs):
    """
    Returns an sklearn Pipeline wrapping StandardScaler + the exact
    RandomForestClassifier hyperparameters used in the MARIDA paper.
    """
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion='gini',
        max_depth=max_depth,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        oob_score=True,
        class_weight='balanced_subsample',
        random_state=5,
        n_jobs=n_jobs,
    )
    return Pipeline([('scaler', StandardScaler()), ('rf', base)])


def _predict_patch(clf, data_path, roi, feature_set, out_dir):
    img, _, _ = load_patch(data_path, roi)
    f = extract_features(img, feature_set, impute=BANDS_MEAN)
    C, H, W = f.shape
    pred = clf.predict(f.reshape(C, -1).T).reshape(H, W).astype(np.uint8)
    np.save(os.path.join(out_dir, f'{roi}.npy'), pred)


# ---------------------------------------------------------------------------
def main(opt):
    fs = opt['features']
    out_dir = opt['out_dir'] or os.path.join(PROJECT_ROOT, 'trained_models', f'rf_{fs}')
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = opt['cache_dir'] or os.path.join(PROJECT_ROOT, 'cache', 'rf_features')

    Xtr, ytr, ctr = build_pixel_table(opt['data_path'], 'train', fs, cache_dir,
                                      opt['n_jobs'], opt['splits_dir'])

    # Reference: weight = 1 / confidence code (High=1.0, Moderate=0.5, Low=1/3).
    # Class imbalance is handled INSIDE the RandomForestClassifier via
    # class_weight='balanced_subsample' -- do NOT add a manual class weight here.
    sw = np.vectorize(CONF_WEIGHT.get)(np.clip(ctr, 1, 3)).astype(np.float64)

    clf = build_rf(opt['n_estimators'], opt['max_depth'], opt['n_jobs'])
    t0 = time.time()
    # Pass sample_weight through the pipeline to the 'rf' step only.
    clf.fit(Xtr, ytr, **{'rf__sample_weight': sw})
    print(f'RF fitted on {len(ytr):,} px in {time.time() - t0:.0f}s')
    joblib.dump(clf, os.path.join(out_dir, 'rf.joblib'), compress=3)

    names = FEATURE_SETS[fs]
    importance = sorted(zip(names, clf.named_steps['rf'].feature_importances_.tolist()),
                        key=lambda t: -t[1])

    results = {'features': fs, 'feature_importance': importance}
    for split in opt['eval_splits']:
        X, y, _ = build_pixel_table(opt['data_path'], split, fs, cache_dir,
                                    opt['n_jobs'], opt['splits_dir'])
        res = per_class_scores(y, clf.predict(X))
        results[split] = res
        print(format_table(res, f'\nRF_{fs.upper()} -- {split} split'))
    print('\nTop features:', ', '.join(f'{n} {v:.3f}' for n, v in importance[:8]))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if opt['predict_patches']:
        pred_dir = os.path.join(out_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        # Pipeline.predict is picklable but the underlying rf n_jobs may
        # trip up joblib's own parallel workers if left at -1; force 1.
        clf.named_steps['rf'].set_params(n_jobs=1)
        rois = read_split(opt['data_path'], 'test', opt['splits_dir'])
        Parallel(n_jobs=opt['n_jobs'], verbose=5)(
            delayed(_predict_patch)(clf, opt['data_path'], r, fs, pred_dir) for r in rois)
        print(f'Saved {len(rois)} full-patch predictions to {pred_dir}')
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_path', default=os.path.join(PROJECT_ROOT, 'data'))
    p.add_argument('--splits_dir', default='splits')
    p.add_argument('--features', choices=list(FEATURE_SETS), default='ss_si_glcm')
    p.add_argument('--n_estimators', type=int, default=125)
    p.add_argument('--max_depth', type=int, default=20)
    p.add_argument('--eval_splits', nargs='+', default=['val', 'test'])
    p.add_argument('--predict_patches', type=str2bool, default=True)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--n_jobs', type=int, default=-1)
    p.add_argument('--seed', type=int, default=5,
                   help='Kept for CLI compatibility; the reference RF always '
                        'uses random_state=5 regardless of this flag.')
    return vars(p.parse_args(argv))


if __name__ == '__main__':
    main(parse_args())