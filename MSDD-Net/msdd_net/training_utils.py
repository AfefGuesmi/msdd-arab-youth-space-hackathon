# -*- coding: utf-8 -*-
"""
training_utils.py

Losses, LR schedulers and optimiser helpers shared by train_marida.py and
train_on_mados.py.

These were previously imported from `train_swin_unetv2.py`, which was never
committed to the repository, so train_on_mados.py could not even be
imported. They now live here as a single, tested source of truth.
"""
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Losses (all honour ignore_index=-1 for unannotated MARIDA/MADOS pixels)
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Multi-class soft Dice loss over annotated pixels only."""

    def __init__(self, num_classes, weight=None, ignore_index=-1, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.register_buffer('weight', weight if weight is not None else torch.ones(num_classes))

    def forward(self, logits, target):
        valid = (target != self.ignore_index)                          # (B,H,W)
        probs = F.softmax(logits, dim=1)                               # (B,C,H,W)
        safe_t = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(safe_t, self.num_classes).permute(0, 3, 1, 2).float()
        v = valid.unsqueeze(1).float()
        probs, onehot = probs * v, onehot * v

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)

        # Only average over classes actually present in this batch, so absent
        # rare classes don't contribute a free "perfect" score.
        present = onehot.sum(dims) > 0
        w = self.weight.to(logits.device) * present.float()
        if w.sum() == 0:
            return logits.sum() * 0.0
        return ((1 - dice) * w).sum() / w.sum()


class CEDiceLoss(nn.Module):
    """(1 - dice_weight) * weighted CE + dice_weight * Dice."""

    def __init__(self, num_classes, weight=None, ignore_index=-1, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, weight=weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (1 - self.dice_weight) * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) with optional per-class alpha weights."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-F.cross_entropy(logits, target, ignore_index=self.ignore_index,
                                        reduction='none'))
        loss = (1 - pt) ** self.gamma * ce
        valid = target != self.ignore_index
        if self.reduction == 'sum':
            return loss[valid].sum()
        if valid.sum() == 0:
            return loss.sum() * 0.0
        return loss[valid].mean()


def build_criterion(loss_type, num_classes, weight=None, focal_gamma=2.0, dice_weight=0.5):
    if loss_type == 'focal':
        return FocalLoss(alpha=weight, gamma=focal_gamma, ignore_index=-1)
    if loss_type == 'dice':
        return DiceLoss(num_classes, weight=weight, ignore_index=-1)
    if loss_type == 'ce_dice':
        return CEDiceLoss(num_classes, weight=weight, ignore_index=-1, dice_weight=dice_weight)
    return nn.CrossEntropyLoss(weight=weight, ignore_index=-1)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, options, steps_per_epoch):
    """
    Returns (warmup_scheduler, main_scheduler, warmup_steps).

    warmup_scheduler is stepped every iteration while global_step <
    warmup_steps; main_scheduler is stepped once per epoch afterwards
    (ReduceLROnPlateau receives the negated validation score).
    """
    warmup_epochs = int(options.get('warmup_epochs', 0) or 0)
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup = None
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    kind = options.get('scheduler', 'multistep')
    epochs = int(options.get('epochs', 100))
    if kind == 'sgdr':
        main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 6), T_mult=2)
    elif kind == 'cosine':
        main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    elif kind == 'plateau':
        main = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:  # multistep -- MARIDA paper: lr x0.1 after epoch 40
        main = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(options.get('lr_steps', [40])), gamma=0.1)
    return warmup, main, warmup_steps


def build_param_groups(model, lr, encoder_lr_mult, weight_decay):
    """Separate (pretrained) encoder params from everything else so the
    encoder can use a smaller LR. Returns (groups, n_encoder, n_other)."""
    enc, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith(('patch_embed.', 'layers.')) else other).append(p)
    groups = [
        {'params': enc, 'lr': lr * encoder_lr_mult, 'weight_decay': weight_decay},
        {'params': other, 'lr': lr, 'weight_decay': weight_decay},
    ]
    return groups, len(enc), len(other)


SWINV2_TINY_URL = ('https://github.com/SwinTransformer/storage/releases/download/'
                   'v2.0.0/swinv2_tiny_patch4_window8_256.pth')


def ensure_pretrained_checkpoint(path=None, cache_dir=None):
    """Return a local path to the ImageNet Swin-V2-T checkpoint,
    downloading it once if `path` is not given/does not exist."""
    if path and os.path.isfile(path):
        return path
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained')
    os.makedirs(cache_dir, exist_ok=True)
    target = path or os.path.join(cache_dir, os.path.basename(SWINV2_TINY_URL))
    if not os.path.isfile(target):
        logging.info("Downloading Swin-V2 ImageNet weights to %s", target)
        torch.hub.download_url_to_file(SWINV2_TINY_URL, target, progress=True)
    return target
