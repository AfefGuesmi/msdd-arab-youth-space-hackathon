# -*- coding: utf-8 -*-
"""Train U-Net (paper recipe) or MSSD-Net (Swin recipe) on MARIDA.

Defaults auto-switch by --model:
  unet -> lr 2e-4, batch 5,  45 epochs, multistep @40, weight_param 1.03
  mssd -> lr 1e-4, batch 8, 150 epochs, cosine + 5 warmup, weight_param 1.02,
          use_pretrained True, patience 25, ce_dice loss
Any explicit flag overrides the model-specific default."""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dataloader import (BANDS_MEAN, BANDS_STD, DATASET_PATH, SPECTRAL_INDEX_NAMES,
                        TEXTURE_FEATURE_NAMES, GenDEBRIS, RandomRotationTransform,
                        gen_weights, marida_class_distribution)
from training_utils import (build_criterion, build_param_groups, build_scheduler,
                            ensure_pretrained_checkpoint)
from utils.cli import int_list, str2bool


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(opt):
    in_ch = opt['input_channels']
    if opt['model'] == 'unet':
        from unet import UNet
        return UNet(input_bands=in_ch, output_classes=opt['num_classes'],
                    hidden_channels=opt['hidden_channels']), 'unet'

    from mssd_net import build_mssd_net
    from utils.config import Config
    cfg = Config()
    swin = cfg.MODEL.SWIN
    model = build_mssd_net(
        opt['variant'], img_size=cfg.DATA.IMG_SIZE, patch_size=swin.PATCH_SIZE,
        in_chans=in_ch, num_classes=opt['num_classes'], embed_dim=swin.EMBED_DIM,
        depths=swin.DEPTHS, num_heads=swin.NUM_HEADS, window_size=swin.WINDOW_SIZE,
        drop_path_rate=swin.DROP_PATH_RATE)
    if opt['use_pretrained']:
        model.load_pretrained(ensure_pretrained_checkpoint(opt['pretrained_path']))
    return model, opt['variant']


def confusion(y_true, y_pred, n):
    return np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)


def scores_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    present = cm.sum(1) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.nan_to_num(tp / (tp + fp + fn))
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    return float(iou[present].mean()), float(f1[present].mean()), iou, f1


@torch.no_grad()
def evaluate(model, loader, device, n_classes, criterion):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    losses = []
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        logits = model(img)
        losses.append(criterion(logits, tgt).item())
        pred = logits.argmax(1)
        valid = tgt != -1
        cm += confusion(tgt[valid].cpu().numpy(), pred[valid].cpu().numpy(), n_classes)
    miou, mf1, iou, f1 = scores_from_cm(cm)
    return float(np.mean(losses)), miou, mf1, iou, f1


def main(opt):
    seed_all(opt['seed'])
    g = torch.Generator()
    g.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aug = [transforms.ToTensor()]
    if opt['train_rotations']:
        aug.append(RandomRotationTransform([-90, 0, 90, 180]))
    if opt['train_hflip']:
        aug.append(transforms.RandomHorizontalFlip())
    std = transforms.Normalize(BANDS_MEAN, BANDS_STD)
    common = dict(standardization=std, path=opt['data_path'], agg_to_water=True,
                  use_spectral_indices=opt['use_spectral_indices'],
                  use_texture_features=opt['use_texture_features'],
                  splits_dir=opt['splits_dir'])
    rare = int_list(opt['rare_classes'])
    train_ds = GenDEBRIS('train', transform=transforms.Compose(aug), rare_classes=rare,
                         copy_paste_prob=opt['copy_paste_prob'],
                         spectral_jitter_prob=opt['spectral_jitter_prob'], **common)
    val_ds = GenDEBRIS('val', transform=transforms.Compose([transforms.ToTensor()]), **common)

    sampler = None
    if opt['oversample_rare']:
        w = train_ds.compute_sample_weights(rare, boost=opt['oversample_boost'])
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    dl_common = dict(
        num_workers=opt['num_workers'],
        pin_memory=opt['pin_memory'] or (device.type == 'cuda'),
        persistent_workers=opt['persistent_workers'] and opt['num_workers'] > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if opt['num_workers'] > 0:
        dl_common['prefetch_factor'] = opt['prefetch_factor']

    train_loader = DataLoader(train_ds, batch_size=opt['batch'],
                              shuffle=sampler is None, sampler=sampler,
                              drop_last=True, **dl_common)
    val_loader = DataLoader(val_ds, batch_size=opt['batch'], shuffle=False, **dl_common)

    model, tag = build_model(opt)
    model.to(device)

    weight = gen_weights(marida_class_distribution(True), c=opt['weight_param']).to(device)
    criterion = build_criterion(opt['loss_type'], opt['num_classes'], weight=weight,
                                focal_gamma=opt['focal_gamma'], dice_weight=opt['dice_weight'])

    if opt['encoder_lr_mult'] != 1.0 and opt['model'] == 'mssd':
        groups, _, _ = build_param_groups(model, opt['lr'], opt['encoder_lr_mult'], opt['decay'])
        optimizer = torch.optim.Adam(groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['decay'])
    warmup, main_sched, warmup_steps = build_scheduler(optimizer, opt, len(train_loader))

    out_dir = os.path.join(opt['out_root'], tag)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(out_dir, 'train.log')),
                                  logging.StreamHandler()], force=True)
    logging.info('Options: %s', json.dumps(opt))
    logging.info('Model %s: %.2fM params', tag,
                 sum(p.numel() for p in model.parameters()) / 1e6)
    with open(os.path.join(out_dir, 'options.json'), 'w') as f:
        json.dump(opt, f, indent=2)

    best, bad, step, history = -1.0, 0, 0, []
    for epoch in range(1, opt['epochs'] + 1):
        model.train()
        t0, losses = time.time(), []
        for img, tgt in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            img, tgt = img.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), tgt)
            loss.backward()
            if opt['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            if warmup is not None and step < warmup_steps:
                warmup.step()
            step += 1
            losses.append(loss.item())

        val_loss, miou, mf1, iou, _ = evaluate(model, val_loader, device,
                                               opt['num_classes'], criterion)
        if step >= warmup_steps:
            if opt['scheduler'] == 'plateau':
                main_sched.step(val_loss)
            else:
                main_sched.step()

        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss,
                            val_miou=miou, val_mf1=mf1, val_md_iou=float(iou[0]),
                            lr=optimizer.param_groups[-1]['lr']))
        logging.info('epoch %d | train %.4f | val %.4f | mIoU %.4f | mF1 %.4f | MD IoU %.4f | %.0fs',
                     epoch, history[-1]['train_loss'], val_loss, miou, mf1, iou[0],
                     time.time() - t0)

        if miou > best:
            best, bad = miou, 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_miou': miou, 'options': opt},
                       os.path.join(out_dir, 'best_model.pth'))
            logging.info('  -> new best (mIoU %.4f) saved', miou)
        else:
            bad += 1
            if opt['patience'] and bad >= opt['patience']:
                logging.info('Early stopping at epoch %d', epoch)
                break

    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    logging.info('Done. Best val mIoU %.4f -> %s', best,
                 os.path.join(out_dir, 'best_model.pth'))
    return best


# Model-specific defaults: only applied if the user did NOT pass the flag.
# Uses argparse.SUPPRESS so we can tell "unset" from "explicitly set".
_MSSD_DEFAULTS = dict(
    lr=1e-4, batch=8, epochs=150, scheduler='cosine', warmup_epochs=5,
    weight_param=1.02, loss_type='ce_dice', use_pretrained=True, patience=25,
    encoder_lr_mult=0.1,
)
_UNET_DEFAULTS = dict(
    lr=2e-4, batch=5, epochs=45, scheduler='multistep', lr_steps=[40],
    warmup_epochs=0, weight_param=1.03, loss_type='ce', use_pretrained=False,
    patience=0, encoder_lr_mult=1.0,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['unet', 'mssd'], default='unet')
    p.add_argument('--variant', default='baseline')
    p.add_argument('--data_path', default=DATASET_PATH)
    p.add_argument('--splits_dir', default='splits')
    p.add_argument('--out_root', default=os.path.join(PROJECT_ROOT, 'trained_models'))
    p.add_argument('--num_classes', type=int, default=11)
    p.add_argument('--hidden_channels', type=int, default=16)
    p.add_argument('--use_spectral_indices', type=str2bool, default=False)
    p.add_argument('--use_texture_features', type=str2bool, default=False)

    # All training hyperparams default to SUPPRESS so we know which the user
    # actually passed; model-specific values are applied afterwards.
    S = argparse.SUPPRESS
    p.add_argument('--epochs', type=int, default=S)
    p.add_argument('--batch', type=int, default=S)
    p.add_argument('--lr', type=float, default=S)
    p.add_argument('--decay', type=float, default=0.0)
    p.add_argument('--scheduler', choices=['multistep', 'cosine', 'sgdr', 'plateau'], default=S)
    p.add_argument('--lr_steps', type=int_list, default=S)
    p.add_argument('--warmup_epochs', type=int, default=S)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--patience', type=int, default=S)
    p.add_argument('--weight_param', type=float, default=S)
    p.add_argument('--loss_type', choices=['ce', 'focal', 'dice', 'ce_dice'], default=S)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--dice_weight', type=float, default=0.5)
    p.add_argument('--encoder_lr_mult', type=float, default=S)
    p.add_argument('--use_pretrained', type=str2bool, default=S)
    p.add_argument('--pretrained_path', default=None)

    p.add_argument('--train_rotations', type=str2bool, default=True)
    p.add_argument('--train_hflip', type=str2bool, default=True)
    p.add_argument('--rare_classes', default='0,2,3,8')
    p.add_argument('--oversample_rare', type=str2bool, default=False)
    p.add_argument('--oversample_boost', type=float, default=5.0)
    p.add_argument('--copy_paste_prob', type=float, default=0.0)
    p.add_argument('--spectral_jitter_prob', type=float, default=0.0)

    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--pin_memory', type=str2bool, default=False)
    p.add_argument('--prefetch_factor', type=int, default=1)
    p.add_argument('--persistent_workers', type=str2bool, default=True)
    p.add_argument('--seed', type=int, default=0)

    opt = vars(p.parse_args(argv))

    # Fill in model-specific defaults for anything the user left unset.
    defaults = _MSSD_DEFAULTS if opt['model'] == 'mssd' else _UNET_DEFAULTS
    for k, v in defaults.items():
        if k not in opt:            # SUPPRESS removes the key entirely when unset
            opt[k] = v

    opt['input_channels'] = 11 \
        + (len(SPECTRAL_INDEX_NAMES) if opt['use_spectral_indices'] else 0) \
        + (len(TEXTURE_FEATURE_NAMES) if opt['use_texture_features'] else 0)
    return opt


if __name__ == '__main__':
    main(parse_args())