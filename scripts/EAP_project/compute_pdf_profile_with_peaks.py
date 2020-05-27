#! /usr/bin/env python
"""
Script to compute EAP.
"""
import argparse

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.shore import ShoreModel

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

    p.add_argument('out_filename',
                   help='Path of the output.')

    p.add_argument('--x', type=int)

    p.add_argument('--y', type=int)

    p.add_argument('--z', type=int)

    p.add_argument('--nb_points', metavar='int', default=15,
                   help='Number of points to sample along the peaks.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=6, type=int,
                   help='Radial order used for the SHORE fit. (Default: 8)')

    p.add_argument('--zeta', metavar='int', default=500, type=int,
                   help='Scale factor.')

    p.add_argument('--lambdaN', metavar='float', default=1e-7, type=float,
                   help='Scale factor.')

    p.add_argument('--lambdaL', metavar='float', default=1e-8, type=float,
                   help='Scale factor.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()

    peaks_data = nib.load(args.peaks)
    peaks = peaks_data.get_fdata()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    shore_model = ShoreModel(gtab, radial_order=args.radial_order,
                             zeta=args.zeta, lambdaN=args.lambdaN,
                             lambdaL=args.lambdaL)

    data = data[args.x, args.y, args.z]
    smfit = shore_model.fit(data)
    voxel = peaks[args.x, args.y, args.z]

    # coef = (smfit._shore_coef).shape

    nb_peaks = int(np.nonzero(voxel)[0].shape[0]/3)
    r_sample = np.linspace(0.008, 0.025, args.nb_points)
    pdf_sample = np.zeros((nb_peaks, args.nb_points))

    for i in range(nb_peaks):
        r, theta, phi = cart2sphere(voxel[3 * i],
                                    voxel[3 * i + 1],
                                    voxel[3 * i + 2])
        theta = np.repeat(theta, args.nb_points)
        phi = np.repeat(phi, args.nb_points)

        r = r * r_sample
        x, y, z = sphere2cart(r, theta, phi)

        r_points = np.vstack((x, y, z)).T

        pdf_sample[i] = smfit.pdf(r_points)

    print(pdf_sample)
    np.savetxt(args.out_filename, pdf_sample, fmt='%1.3f')


if __name__ == "__main__":
    main()