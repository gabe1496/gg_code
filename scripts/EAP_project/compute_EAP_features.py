#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute EAP.
"""
import argparse

import nibabel as nib
import numpy as np

from gg_code.EAP_utils import glyph_from_model, peaks

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mapmri import MapmriModel
from dipy.io.streamline import load_tractogram

from scilpy.utils.bvec_bval_tools import check_b0_threshold
# from scilpy.reconst.shore_ozarslan import ShoreOzarslanModel


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('roi',
                   help='Path of the region of interest.')

    p.add_argument('out_directory',
                   help='Path of the output directory.')

    p.add_argument('--big_delta', metavar='float', default=None, type=float,
                   help='Big delta for gtab.')

    p.add_argument('--small_delta', metavar='float', default=None, type=float,
                   help='Small delta for gtab.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=6, type=int,
                   help='Radial order used for the SHORE fit. (Default: 6)')

    p.add_argument('--anisotropic_scaling', metavar='bool', default=True,
                   help='Anisotropique scaling.')

    p.add_argument('--pos_const', metavar='bool', default=True,
                   help='Positivity constraint.')

    p.add_argument('--lap_reg',metavar='int', default=1,
                   help='Laplacian regularization.')

    p.add_argument('--lap_weight', metavar='float', default=0.2,
                   help='Laplacian weighting in case of laplacian regularization.')

    p.add_argument('--sphere', default='repulsion724',
                   help='Type of sphere for the pdf compute.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    if args.big_delta is not None and args.small_delta is not None:
        gtab = gradient_table(bvals,
                              bvecs,
                              b0_threshold=bvals.min(),
                              big_delta=args.big_delta,
                              small_delta=args.small_delta)
    else:
        gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    sphere = get_sphere(args.sphere)

    # Fit the model
    if args.lap_reg == 1:
        mapmri_model = MapmriModel(gtab, radial_order=args.radial_order,
                                   anisotropic_scaling=args.anisotropic_scaling,
                                   laplacian_regularization=True,
                                   laplacian_weighting=args.lap_weight,
                                   positivity_constraint=args.pos_const)

    else:
        mapmri_model = MapmriModel(gtab, radial_order=args.radial_order,
                                   anisotropic_scaling=args.anisotropic_scaling,
                                   laplacian_regularization=False,
                                   positivity_constraint=args.pos_const)

    odf = glyph_from_model.compute_odf_mapmri(mapmri_model, data_small, sphere, s=4)

    mapmri_fit = mapmri_model.fit(data)
    rtop = mapmri_fit.rtop()
    rtap = mapmri_fit.rtap()
    rtpp = mapmri_fit.rtpp()
    nib.save(nib.Nifti1Image(odf, affine), args.out_directory + '_odf.nii.gz')
    nib.save(nib.Nifti1Image(rtop, affine), args.out_directory + '_rtop.nii.gz')
    nib.save(nib.Nifti1Image(rtap, affine), args.out_directory + '_rtap.nii.gz')
    nib.save(nib.Nifti1Image(rtpp, affine), args.out_directory + '_rtpp.nii.gz')


if __name__ == "__main__":
    main()
