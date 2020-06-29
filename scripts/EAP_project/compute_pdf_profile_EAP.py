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

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('roi',
                   help='Path of the region of interest.')

    p.add_argument('out_directory',
                   help='Path of the output directory.')

    p.add_argument('--nb_points', metavar='int', default=15,
                   help='Number of points to sample along the peaks.')

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

    # Load ROI
    roi = nib.load(args.roi)
    mask = roi.get_fdata()

    # Segment data from roi
    ind_mask = np.argwhere(mask > 0)
    data_small = data[np.min(ind_mask[:, 0]):np.max(ind_mask[:, 0]) + 1,
                      np.min(ind_mask[:, 1]):np.max(ind_mask[:, 1]) + 1,
                      np.min(ind_mask[:, 2]):np.max(ind_mask[:, 2]) + 1]

    del data

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

    mapmri_fit = mapmri_model.fit(data_small)

    sphere = get_sphere(args.sphere)

    npeaks = 5

    odf = mapmri_fit.odf(sphere)

    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    for idx in ndindex(shape):
        direction, pk, ind = peak_directions(odf[idx], sphere)
        n = min(npeaks, pk.shape[0])
        peaks_dirs[idx][:n] = direction[:n]
        peaks_odf = reshape_peaks_for_visualization(peaks_dirs)
    nib.save(nib.Nifti1Image(peaks_odf, affine), args.out_directory + 'peaks_eap_odf.nii.gz')

    nib.save(nib.Nifti1Image(odf, affine), args.out_directory + 'eap_odf.nii.gz')

    peaks_cc = np.zeros_like(peaks_odf)
    peaks_af = np.zeros_like(peaks_odf)
    peaks_pt = np.zeros_like(peaks_odf)

    for ind in ind_mask:
        peak_cc = peaks_odf[ind[0], ind[1], ind[2]]
        peak_cc = peak_cc.reshape(5, 3)

        ind_cc = np.argwhere(np.argmax(np.abs(peak_cc), axis=1) < 1)
        if (ind_cc.size) == 0:
            new_peak_cc = np.zeros((15))
            peaks_cc[ind[0], ind[1], ind[2]] = new_peak_cc
        else:
            new_peak_cc = np.zeros((15))
            new_peak_cc[0:3] = peak_cc[ind_cc[0]]

            peaks_cc[ind[0], ind[1], ind[2]] = new_peak_cc

        peak_af = peaks_odf[ind[0], ind[1], ind[2]]
        peak_af = peak_af.reshape(5, 3)
        ind_af = np.argwhere(np.logical_and(np.argmax(np.abs(peak_af), axis=1) < 2,
                                            np.argmax(np.abs(peak_af), axis=1) > 0))

        if (ind_af.size) == 0:
            new_peak_af = np.zeros((15))
            peaks_af[ind[0], ind[1], ind[2]] = new_peak_af
        else:
            new_peak_af = np.zeros((15))
            new_peak_af[0:3] = peak_af[ind_af[0]]

            peaks_af[ind[0], ind[1], ind[2]] = new_peak_af

        peak_pt = peaks_odf[ind[0], ind[1], ind[2]]
        peak_pt = peak_pt.reshape(5, 3)
        ind_pt = np.argwhere(np.argmax(np.abs(peak_pt), axis=1) > 1)

        if (ind_pt.size) == 0:
            new_peak_pt = np.zeros((15))
            peaks_pt[ind[0], ind[1], ind[2]] = new_peak_pt
        else:
            new_peak_pt = np.zeros((15))
            new_peak_pt[0:3] = peak_pt[ind_pt[0]]
            peaks_pt[ind[0], ind[1], ind[2]] = new_peak_pt

    # Save peaks file depending on the bundle
    nib.save(nib.Nifti1Image(peaks_cc, affine), args.out_directory + 'peaks_cc.nii.gz')
    nib.save(nib.Nifti1Image(peaks_af, affine), args.out_directory + 'peaks_af.nii.gz')
    nib.save(nib.Nifti1Image(peaks_pt, affine), args.out_directory + 'peaks_pt.nii.gz')

    list_vox = np.indices((peaks_odf.shape[0],
                           peaks_odf.shape[1],
                           peaks_odf.shape[2])).T.reshape(-1, 3)

    r_sample = np.linspace(0.008, 0.025, args.nb_points)
    pdf_sample_cc = np.zeros((list_vox.shape[0], args.nb_points))
    pdf_sample_af = np.zeros((list_vox.shape[0], args.nb_points))
    pdf_sample_pt = np.zeros((list_vox.shape[0], args.nb_points))
    counter = 0

    for vox in list_vox:
        peak_cc = peaks_cc[vox[0], vox[1], vox[2]]
        data = data_small[vox[0], vox[1], vox[2]]
        mapmri_fit = mapmri_model.fit(data)

        if np.max(np.abs(peak_cc)) < 0.001:
            pdf_sample_cc = np.delete(pdf_sample_cc, counter, 0)

        else:
            r, theta, phi = cart2sphere(peak_cc[0], peak_cc[1], peak_cc[2])
            theta = np.repeat(theta, args.nb_points)
            phi = np.repeat(phi, args.nb_points)

            x, y, z = sphere2cart(r_sample, theta, phi)

            r_points = np.vstack((x, y, z)).T

            pdf_sample_cc[counter] = mapmri_fit.pdf(r_points)
            counter += 1

    np.savetxt(args.out_directory + 'odf_pdf_cc.csv', pdf_sample_cc, fmt='%1.3f', delimiter=',')

    counter = 0

    for vox in list_vox:
        peak_af = peaks_af[vox[0], vox[1], vox[2]]
        data = data_small[vox[0], vox[1], vox[2]]
        mapmri_fit = mapmri_model.fit(data)

        if np.max(np.abs(peak_af)) < 0.001:
            pdf_sample_af = np.delete(pdf_sample_af, counter, 0)

        else:
            r, theta, phi = cart2sphere(peak_af[0], peak_af[1], peak_af[2])
            theta = np.repeat(theta, args.nb_points)
            phi = np.repeat(phi, args.nb_points)

            x, y, z = sphere2cart(r_sample, theta, phi)

            r_points = np.vstack((x, y, z)).T

            pdf_sample_af[counter] = mapmri_fit.pdf(r_points)
            counter += 1

    np.savetxt(args.out_directory + 'odf_pdf_af.csv', pdf_sample_af, fmt='%1.3f', delimiter=',')

    counter = 0

    for vox in list_vox:
        peak_pt = peaks_pt[vox[0], vox[1], vox[2]]
        data = data_small[vox[0], vox[1], vox[2]]
        mapmri_fit = mapmri_model.fit(data)

        if np.max(np.abs(peak_pt)) < 0.001:
            pdf_sample_pt = np.delete(pdf_sample_pt, counter, 0)

        else:
            r, theta, phi = cart2sphere(peak_pt[0], peak_pt[1], peak_pt[2])
            theta = np.repeat(theta, args.nb_points)
            phi = np.repeat(phi, args.nb_points)

            x, y, z = sphere2cart(r_sample, theta, phi)

            r_points = np.vstack((x, y, z)).T

            pdf_sample_pt[counter] = mapmri_fit.pdf(r_points)
            counter += 1

    np.savetxt(args.out_directory + 'odf_pdf_pt.csv', pdf_sample_pt, fmt='%1.3f', delimiter=',')


if __name__ == "__main__":
    main()
