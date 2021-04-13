#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.mapmri import MapmriModel
from dipy.core.ndindex import ndindex
from dipy.core.geometry import sphere2cart, cart2sphere

from scilpy.utils.bvec_bval_tools import check_b0_threshold


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('peaks',
                   help='Peaks.')

    p.add_argument('out_directory',
                   help='Path of the output directory.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=8, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')

    p.add_argument('--anisotropic_scaling', metavar='bool', default=True,
                   help='Anisotropique scaling.')

    p.add_argument('--pos_const', metavar='bool', default=True,
                   help='Positivity constraint.')

    p.add_argument('--lap_reg',metavar='int', default=1,
                   help='Laplacian regularization.')

    p.add_argument('--lap_weight', metavar='float', default=0.2,
                   help='Laplacian weighting in case of laplacian regularization.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_points = 25

    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.get_affine()

    peaks_data = nib.load(args.peaks)
    peaks = peaks_data.get_fdata()

    peak = peaks[0:3]

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

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

    mapmri_fit = mapmri_model.fit(data)
    r_sample = np.linspace(0.0, 0.025, nb_points)

    pdf = np.zeros((nb_points))
    r, theta, phi = cart2sphere(peak[0], peak[1], peak[2])
    theta = np.repeat(theta, nb_points)
    phi = np.repeat(phi, nb_points)

    x, y, z = sphere2cart(r_sample, theta, phi)

    r_points = np.vstack((x, y, z)).T

    pdf = mapmri_fit.pdf(r_points)

    nib.save(nib.Nifti1Image(pdf, affine), args.out_directory + 'eap_pdf_profile.nii.gz')
    nib.save(nib.Nifti1Image(r_points, affine), args.out_directory + 'eap_points_profile.nii.gz')


if __name__ == "__main__":
    main()
