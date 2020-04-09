#! /usr/bin/env python
"""
Script to visualize pdf from EAP.
"""
import argparse

import nibabel as nib
import numpy as np

from fury import window, actor

from dipy.data import get_sphere


WINDOW_SIZE = (600, 600)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_pdf',
                   help='Path of the input pdf.')
    p.add_argument('sphere',
                   help='Type of sphere to use.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    pdf_data = nib.load(args.in_pdf)
    pdf = pdf_data.get_fdata()

    sph = get_sphere(args.sphere)

    ren = window.renderer(background=window.colors.black)

    pdf_actor = actor.odf_slicer(pdf, sphere=sph, colormap='jet', scale=0.4)

    ren.add(pdf_actor)
    window.show(ren, size=WINDOW_SIZE)

    ren.rm(pdf_actor)
    window.rm_all(ren)


if __name__ == "__main__":
    main()
