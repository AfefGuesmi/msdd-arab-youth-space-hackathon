# -*- coding: utf-8 -*-
"""Small argparse helpers shared by every script.

`type=bool` is a classic argparse trap: bool("False") is True, so
`--tta False` silently ENABLES TTA. Use `str2bool` instead.
"""
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if v in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def int_list(value):
    """'45,65' -> [45, 65] (argparse `type=list` splits into characters)."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).strip('[] ').split(',') if v.strip()]
