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

    p.add_argument('cc',
                   help='Path of the cc bundle.')

    p.add_argument('af',
                   help='Path of the af bundle.')

    p.add_argument('pt',
                   help='Path of the pt bundle.')

    p.add_argument('out_directory',
                   help='Path of the output directory.')

    p.add_argument('--nb_points', metavar='int', default=20,
                   help='Number of points to sample along the peaks.')

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
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Load ROI
    roi = nib.load(args.roi)
    mask = roi.get_fdata()

    # Load bundles
    sft_cc = load_tractogram(args.cc, 'same', bbox_valid_check=True)
    sft_af = load_tractogram(args.af, 'same', bbox_valid_check=True)
    sft_pt = load_tractogram(args.pt, 'same', bbox_valid_check=True)
    sft_cc.to_vox()
    sft_af.to_vox()
    sft_pt.to_vox()

    # Segment data from roi
    ind_mask = np.argwhere(mask > 0)
    data_small = data[np.min(ind_mask[:, 0]):np.max(ind_mask[:, 0]) + 1,
                      np.min(ind_mask[:, 1]):np.max(ind_mask[:, 1]) + 1,
                      np.min(ind_mask[:, 2]):np.max(ind_mask[:, 2]) + 1]

    sphere = get_sphere(args.sphere)
    r = 0.015

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

    # pdf = glyph_from_model.compute_pdf(mapmri_model, data_small, sphere, r)
    # print('PDF done.')

    # all_peaks = peaks.compute_peaks(pdf, sphere)
    # print('Peaks done.')

    # nib.save(nib.Nifti1Image(all_peaks, affine), args.out_directory + 'peaks_eap_pdf.nii.gz')
    # nib.save(nib.Nifti1Image(pdf, affine), args.out_directory + 'eap_pdf.nii.gz')

    pdf = nib.load(args.out_directory + 'eap_pdf.nii.gz')
    pdf = pdf.get_fdata()
    all_peaks = nib.load(args.out_directory + 'peaks_eap_pdf.nii.gz')
    all_peaks = all_peaks.get_fdata()

    cc_avr, cc_peaks = peaks.segment_peaks_from_bundle(all_peaks, sft_cc, mask, args.sphere)
    # af_avr, af_peaks = peaks.segment_peaks_from_bundle(all_peaks, sft_af, mask, args.sphere)
    # pt_avr, pt_peaks = peaks.segment_peaks_from_bundle(all_peaks, sft_pt, mask, args.sphere)

    # Save peaks file depending on the bundle
    nib.save(nib.Nifti1Image(cc_avr.astype('float32'), affine), args.out_directory + 'avr_cc.nii.gz')
    # nib.save(nib.Nifti1Image(af_avr.astype('float32'), affine), args.out_directory + 'avr_af.nii.gz')
    # nib.save(nib.Nifti1Image(pt_avr.astype('float32'), affine), args.out_directory + 'avr_pt.nii.gz')

    nib.save(nib.Nifti1Image(cc_peaks.astype('float32'), affine), args.out_directory + 'peaks_cc.nii.gz')
    # nib.save(nib.Nifti1Image(af_peaks.astype('float32'), affine), args.out_directory + 'peaks_af.nii.gz')
    # nib.save(nib.Nifti1Image(pt_peaks.astype('float32'), affine), args.out_directory + 'peaks_pt.nii.gz')

    print('Segmentation done.')

    # pdf_sample_cc = peaks.compute_bundle_eap_profile_along_peaks(model, data_small, cc_peaks)
    # pdf_sample_af = peaks.compute_bundle_eap_profile_along_peaks(model, data_small, af_peaks)
    # pdf_sample_pt = peaks.compute_bundle_eap_profile_along_peaks(model, data_small, pt_peaks)

    # np.savetxt(args.out_directory + 'cc_pdf_profile.csv', pdf_sample_cc, fmt='%1.3f', delimiter=',')
    # np.savetxt(args.out_directory + 'af_pdf_profile.csv', pdf_sample_af, fmt='%1.3f', delimiter=',')
    # np.savetxt(args.out_directory + 'pt_pdf_profile.csv', pdf_sample_pt, fmt='%1.3f', delimiter=',')


if __name__ == "__main__":
    main()
