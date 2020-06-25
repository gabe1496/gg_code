#! /usr/bin/env python
"""
Script to compute EAP.
"""
import argparse

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.core.ndindex import ndindex
from dipy.io.gradients import read_bvals_bvecs
from dipy.direction.peaks import (peak_directions,
                                  reshape_peaks_for_visualization)
from dipy.reconst.mapmri import MapmriModel

from scilpy.utils.bvec_bval_tools import check_b0_threshold
# from scilpy.reconst.shore_ozarslan import ShoreOzarslanModel


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    # p.add_argument('mask',
    #                help='Path of the mask.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('out_filename',
                   help='Path of the output pdf.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=8, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')

    p.add_argument('--anisotropic_scaling', metavar='bool', default=True,
                   help='Anisotropique scaling.')

    p.add_argument('--pos_const', metavar='bool', default=True,
                   help='Positivity constraint.')

    p.add_argument('--lap_reg', metavar='bool', default=True,
                   help='Laplacian regularization.')

    p.add_argument('--lap_weight', metavar='float', default=0.2,
                   help='Laplacian weighting in case of laplacian regularization.')

    p.add_argument('--sphere', default='repulsion724',
                   help='Type of sphere for the pdf compute.')

    p.add_argument('--radii', metavar='float', default=0.015, type=float,
                   help='The radii for which to compute pdf.')

    p.add_argument('--peaks', metavar='file', default='',
                   help='Output filename for the extracted peaks.')

    p.add_argument('--odf', metavar='file', default='',
                   help='Output filename for the odf.')

    p.add_argument('--odf_peaks', metavar='file', default='',
                   help='Output filename for the odf peaks.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    shape = data.shape[:-1]

    # Fit the model
    if args.lap_reg:
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
    del data

    sphere_rone = get_sphere(args.sphere)
    vertices = sphere_rone.vertices
    r, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    r = r * args.radii
    x, y, z = sphere2cart(r, theta, phi)

    vertices_new = np.vstack((x, y, z)).T
    mapmri_pdf = mapmri_fit.pdf(vertices_new)

    npeaks = 5

    if args.peaks:
        peaks_dirs = np.zeros((shape + (npeaks, 3)))
        for idx in ndindex(shape):
            direction, pk, ind = peak_directions(mapmri_pdf[idx], sphere_rone)
            n = min(npeaks, pk.shape[0])
            peaks_dirs[idx][:n] = direction[:n]
        nib.save(nib.Nifti1Image(
            reshape_peaks_for_visualization(peaks_dirs), affine),
            args.peaks)

    nib.save(nib.Nifti1Image(mapmri_pdf, affine), args.out_filename)

    if args.odf:
        odf = mapmri_fit.odf(sphere_rone)

        if args.peaks:
            peaks_dirs = np.zeros((shape + (npeaks, 3)))
            for idx in ndindex(shape):
                direction, pk, ind = peak_directions(odf[idx], sphere_rone)
                n = min(npeaks, pk.shape[0])
                peaks_dirs[idx][:n] = direction[:n]
            nib.save(nib.Nifti1Image(
                reshape_peaks_for_visualization(peaks_dirs), affine),
                args.odf_peaks)

        nib.save(nib.Nifti1Image(odf, affine), args.odf)


if __name__ == "__main__":
    main()
