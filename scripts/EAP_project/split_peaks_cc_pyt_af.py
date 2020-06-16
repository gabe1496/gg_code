#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_peaks_cc',
                   help='Path of the input file peaks for the CC.')

    p.add_argument('in_peaks_af',
                   help='Path of the input file peaks for the AF.')

    p.add_argument('in_peaks_pt',
                   help='Path of the input file peaks for the PYT.')

    p.add_argument('roi',
                   help='Path of the region of interest.')

    p.add_argument('out_directory',
                   help='Path of the directory.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load peaks data
    data_cc = nib.load(args.in_peaks_cc)
    peaks_cc = data_cc.get_fdata()
    affine_cc = data_cc.affine

    data_af = nib.load(args.in_peaks_af)
    peaks_af = data_af.get_fdata()
    affine_af = data_af.affine

    data_pt = nib.load(args.in_peaks_pt)
    peaks_pt = data_pt.get_fdata()
    affine_pt = data_pt.affine

    # Load ROI
    roi = nib.load(args.roi)
    mask = roi.get_fdata()
    ind_mask = np.argwhere(mask > 0)
    grid = np.indices((np.max(ind_mask[:, 0]) - np.min(ind_mask[:, 0]) + 1,
                       np.max(ind_mask[:, 1]) - np.min(ind_mask[:, 1]) + 1,
                       np.max(ind_mask[:, 2]) - np.min(ind_mask[:, 2]) + 1)).T.reshape(-1,3)
    grid_shape = grid.shape[0]
    grid_data = np.vstack((np.repeat(np.min(ind_mask[:, 0]), grid_shape),
                           np.repeat(np.min(ind_mask[:, 1]), grid_shape),
                           np.repeat(np.min(ind_mask[:, 2]), grid_shape))).T

    ind_square_mask = grid_data + grid

    print(np.max(ind_mask[:, 0]))
    print(ind_square_mask)

    # Split peaks for the three bundles
    for ind in ind_mask:
        peak_cc = peaks_cc[ind[0], ind[1], ind[2]]
        peak_cc = peak_cc.reshape(5, 3)
        ind_cc = np.argwhere(np.argmax(peak_cc, axis=1) < 1)
        if (ind_cc.size) == 0:
            new_peak_cc = np.zeros((15))
            peaks_cc[ind[0], ind[1], ind[2]] = new_peak_cc
        else:
            new_peak_cc = np.zeros((15))
            new_peak_cc[0:3] = peak_cc[ind_cc[0]]

            peaks_cc[ind[0], ind[1], ind[2]] = new_peak_cc

        peak_af = peaks_af[ind[0], ind[1], ind[2]]
        peak_af = peak_af.reshape(5, 3)
        ind_af = np.argwhere(np.logical_and(np.argmax(peak_af, axis=1) < 2,
                                            np.argmax(peak_af, axis=1) > 0))

        if (ind_af.size) == 0:
            new_peak_af = np.zeros((15))
            peaks_af[ind[0], ind[1], ind[2]] = new_peak_af
        else:
            new_peak_af = np.zeros((15))
            new_peak_af[0:3] = peak_af[ind_af[0]]

            peaks_af[ind[0], ind[1], ind[2]] = new_peak_af

        peak_pt = peaks_pt[ind[0], ind[1], ind[2]]
        peak_pt = peak_pt.reshape(5, 3)
        ind_pt = np.argwhere(np.argmax(peak_pt, axis=1) > 1)

        if (ind_pt.size) == 0:
            new_peak_pt = np.zeros((15))
            peaks_pt[ind[0], ind[1], ind[2]] = new_peak_pt
        else:
            new_peak_pt = np.zeros((15))
            new_peak_pt[0:3] = peak_pt[ind_cc[0]]

            peaks_pt[ind[0], ind[1], ind[2]] = new_peak_pt

    # Save peaks file depending on the bundle
    nib.save(nib.Nifti1Image(peaks_cc, affine_cc), args.out_directory + 'peaks_cc.nii.gz')
    nib.save(nib.Nifti1Image(peaks_af, affine_af), args.out_directory + 'peaks_af.nii.gz')
    nib.save(nib.Nifti1Image(peaks_pt, affine_pt), args.out_directory + 'peaks_pt.nii.gz')


if __name__ == "__main__":
    main()
