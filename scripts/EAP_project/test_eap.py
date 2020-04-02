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
from dipy.io.gradients import read_bvals_bvecs

from scilpy.reconst.shore_ozarslan import ShoreOzarslanModel
from scilpy.utils.bvec_bval_tools import check_b0_threshold


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')
    
    p.add_argument('mask',
                   help='Path of the mask.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=8, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')


    p.add_argument('--regul_weighting', action='store', dest='regul_weighting',
                   metavar='float', default=0.2, type=float,
                   help='Laplacian weighting for the regularization. '
                        '0.0 will make the generalized cross-validation ' +
                        '(GCV) kick in. (Default: 0.2)')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    mask = nib.load(args.mask).get_data().astype(np.bool)
    voxels_with_values_mask = data[:, :, :, 0] > 0
    mask = voxels_with_values_mask * mask

    sphere = get_sphere('symetric724')

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

    smfit = shore_model.fit(data, mask)

    coeff = smfit.shore_coeff()
    print(coeff.shape)


if __name__ == "__main__":
    main()
