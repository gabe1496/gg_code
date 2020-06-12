#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.core.ndindex import ndindex
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.direction.peaks import (peak_directions,
                                  reshape_peaks_for_visualization)

from scilpy.io.utils import add_sh_basis_args
from scilpy.reconst.utils import find_order_from_nb_coeff


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_SH_object',
                   help='Path of the input SH object.')

    p.add_argument('out_filename',
                   help='Path of the output peaks.')

    p.add_argument('--sphere', default='repulsion724',
                   help='Type of sphere for the pdf compute.')

    add_sh_basis_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    sphere = get_sphere(args.sphere)

    obj = nib.load(args.in_SH_object)
    data = obj.get_fdata()
    affine = obj.affine

    shape = data.shape[:-1]
    sh_shape = data.shape
    sh_order = find_order_from_nb_coeff(sh_shape)
    npeaks = 5
    sf = sh_to_sf(data, sphere, basis_type=args.sh_basis, sh_order=sh_order)

    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    for idx in ndindex(shape):
        direction, pk, ind = peak_directions(sf[idx], sphere)
        n = min(npeaks, pk.shape[0])
        peaks_dirs[idx][:n] = direction[:n]
    nib.save(nib.Nifti1Image(
        reshape_peaks_for_visualization(peaks_dirs), affine),
        args.out_filename)


if __name__ == "__main__":
    main()
