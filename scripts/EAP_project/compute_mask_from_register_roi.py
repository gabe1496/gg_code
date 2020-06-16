#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('roi',
                   help='Path of the region of interest.')

    p.add_argument('out_filename',
                   help='Path of the output peaks.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load roi
    data = nib.load(args.roi)
    roi = data.get_fdata()
    affine = data.affine

    # Mask
    mask = np.where(roi > 0.0, 1.0, 0.0)

    # Save it
    nib.save(nib.Nifti1Image(mask, affine), args.out_filename)


if __name__ == "__main__":
    main()
