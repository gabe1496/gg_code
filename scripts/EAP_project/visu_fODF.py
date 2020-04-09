#! /usr/bin/env python
"""
Script to visualize fODF.
"""
import argparse

import nibabel as nib
import numpy as np

from fury import window, actor

from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from scilpy.io.utils import add_sh_basis_args

WINDOW_SIZE = (600, 600)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fodf',
                   help='Path of the input fodf.')
    p.add_argument('sphere',
                   help='Type of sphere to use.')

    add_sh_basis_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    fodf_data = nib.load(args.in_fodf)
    fodf = fodf_data.get_fdata()

    sph = get_sphere(args.sphere)
    fodf_sf = sh_to_sf(fodf, sph, sh_order=8, basis_type=args.sh_basis)

    ren = window.renderer(background=window.colors.black)

    sf_actor = actor.odf_slicer(fodf_sf, sphere=sph, colormap='jet', scale=0.4)

    ren.add(sf_actor)
    window.show(ren, size=WINDOW_SIZE)

    ren.rm(sf_actor)
    window.rm_all(ren)


if __name__ == "__main__":
    main()
