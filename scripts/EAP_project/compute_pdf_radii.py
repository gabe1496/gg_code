#! /usr/bin/env python
"""
Script to compute EAP.
"""
import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.core.ndindex import ndindex
from dipy.io.gradients import read_bvals_bvecs
from dipy.direction.peaks import (peak_directions,
                                  reshape_peaks_for_visualization)

from scilpy.reconst.shore_ozarslan import ShoreOzarslanModel
from scilpy.utils.bvec_bval_tools import check_b0_threshold


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('mask',
                   help='Path of the mask.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('out_filename',
                   help='Path of the output.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=8, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')

    p.add_argument('--regul_weighting', action='store', dest='regul_weighting',
                   metavar='float', default=0.2, type=float,
                   help='Laplacian weighting for the regularization. '
                        '0.0 will make the generalized cross-validation ' +
                        '(GCV) kick in. (Default: 0.2)')

    p.add_argument('--sphere', default='repulsion724',
                   help='Type of sphere for the pdf compute.')

    p.add_argument('--radii', metavar='float', default=0.015, type=float,
                   help='The radii for which to compute pdf.')

    p.add_argument('--peaks', metavar='file', default='',
                   help='Output filename for the extracted peaks.')

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

    mask = nib.load(args.mask).get_data().astype(np.bool)
    voxels_with_values_mask = data[:, :, :, 0] > 0
    mask = voxels_with_values_mask * mask

    sphere_rone = get_sphere(args.sphere)
    vertices = sphere_rone.vertices
    r, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    r = r * args.radii
    x, y, z = sphere2cart(r, theta, phi)

    vertices_new = np.vstack((x, y, z)).T

    if args.regul_weighting <= 0:
        logging.info('Now computing SHORE ODF of radial order {0}'
                     .format(args.radial_order) +
                     ' and Laplacian generalized cross-validation')

        shore_model = ShoreOzarslanModel(gtab, radial_order=args.radial_order,
                                         laplacian_regularization=True,
                                         laplacian_weighting='GCV')
    else:
        logging.info('Now computing SHORE ODF of radial order {0}'
                     .format(args.radial_order) +
                     ' and Laplacian regularization weight of {0}'
                     .format(args.regul_weighting))

        shore_model = ShoreOzarslanModel(gtab, radial_order=args.radial_order,
                                         laplacian_regularization=True,
                                         laplacian_weighting=args.regul_weighting)

    data = data[65:68, 81:84, 32:35]
    shape = data.shape[:-1]
    npeaks = 5
    smfit = shore_model.fit(data)
    pdf = smfit.pdf(vertices_new)

    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    for idx in ndindex(shape):
        direction, pk, ind = peak_directions(pdf[idx], sphere_rone)
        n = min(npeaks, pk.shape[0])
        peaks_dirs[idx][:n] = direction[:n]

    if args.peaks:
        nib.save(nib.Nifti1Image(
            reshape_peaks_for_visualization(peaks_dirs), affine),
            args.peaks)

    nib.save(nib.Nifti1Image(pdf, affine), args.out_filename)


if __name__ == "__main__":
    main()
