#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np

from dipy.reconst.mapmri import MapmriModel
from dipy.core.gradients import gradient_table
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.io.gradients import read_bvals_bvecs

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

    p.add_argument('in_peaks',
                   help='Path of the input file peaks.')

    p.add_argument('roi',
                   help='Path of the region of interest.')

    p.add_argument('out_filename',
                   help='Path of the output peaks.')

    p.add_argument('--nb_points', metavar='int', default=15,
                   help='Number of points to sample along the peaks.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=6, type=int,
                   help='Radial order used for the SHORE fit. (Default: 6)')

    p.add_argument('--pos_const', metavar='bool', default=True,
                   help='Positivity constraint.')

    p.add_argument('--lap_reg', metavar='bool', default=True,
                   help='Laplacian regularization.')

    p.add_argument('--lap_weight', metavar='float', default=0.2,
                   help='Laplacian weighting in case of laplacian regularization.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load data, bvals, bvecs
    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.get_affine()

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Load ROI
    roi = nib.load(args.roi)
    mask = roi.get_fdata()

    # Segment data from roi
    ind_mask = np.argwhere(mask > 0)
    data_small = data[np.min(ind_mask[:, 0]):np.max(ind_mask[:, 0]) + 1,
                      np.min(ind_mask[:, 1]):np.max(ind_mask[:, 1]) + 1,
                      np.min(ind_mask[:, 2]):np.max(ind_mask[:, 2]) + 1]

    del data

    # Load peaks
    in_peaks = nib.load(args.in_peaks)
    peaks = in_peaks.get_fdata()
    peaks_small = peaks[np.min(ind_mask[:, 0]):np.max(ind_mask[:, 0]) + 1,
                        np.min(ind_mask[:, 1]):np.max(ind_mask[:, 1]) + 1,
                        np.min(ind_mask[:, 2]):np.max(ind_mask[:, 2]) + 1]

    # Fit the model
    if args.lap_reg:
        mapmri_model = MapmriModel(gtab, radial_order=args.radial_order,
                                   laplacian_regularization=True,
                                   laplacian_weighting=args.lap_weight,
                                   positivity_constraint=args.pos_const)

    else:
        mapmri_model = MapmriModel(gtab, radial_order=args.radial_order,
                                   laplacian_regularization=False,
                                   positivity_constraint=args.pos_const)

    # Compute pdf profile for each voxel and save
    # nb_vox = peaks.shape[0] * peaks.shape[1] * peaks.shape[2]
    # peaks_list = peaks_small.reshape((int(peaks_small/3), -1))
    # print(peaks_list.shape)
    list_vox = np.indices((peaks_small.shape[0], peaks_small.shape[1], peaks_small.shape[2])).T.reshape(-1,3)

    r_sample = np.linspace(0.008, 0.025, args.nb_points)
    pdf_sample = np.zeros((list_vox.shape[0], args.nb_points))
    counter = 0
    print(list_vox.shape[0])

    for vox in list_vox:
        peak = peaks_small[vox[0], vox[1], vox[2]]
        data = data_small[vox[0], vox[1], vox[2]]
        mapmri_fit = mapmri_model.fit(data)
        
        if np.abs(np.max(peak)) < 0.001:
            counter += 1
        else:
            r, theta, phi = cart2sphere(peak[0], peak[1], peak[2])
            theta = np.repeat(theta, args.nb_points)
            phi = np.repeat(phi, args.nb_points)

            x, y, z = sphere2cart(r_sample, theta, phi)

            r_points = np.vstack((x, y, z)).T

            pdf_sample[counter] = mapmri_fit.pdf(r_points)
            counter += 1

    np.savetxt(args.out_filename, pdf_sample, fmt='%1.3f')


if __name__ == "__main__":
    main()
