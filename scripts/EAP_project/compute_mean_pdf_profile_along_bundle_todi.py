#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np

from dipy.reconst.mapmri import MapmriModel
from dipy.core.gradients import gradient_table
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.utils.bvec_bval_tools import check_b0_threshold
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Path of the bundle file.')

    p.add_argument('in_label_map',
                   help='Path of the input label map.')

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('out_directory',
                   help='Path of the output directory.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the EAP values according to '
                        'segment lengths. [%(default)s]')

    p.add_argument('--nb_points', metavar='int', default=20,
                   help='Number of points to sample along the peaks.')

    p.add_argument('--nb_sections', metavar='int', default=5,
                   help='Number of sections dividing the bundle.')

    p.add_argument('--sample_size', metavar='int', default=20,
                   help='Number of points to sample in each section of the bundle.')

    p.add_argument('--radial_order', action='store', dest='radial_order',
                   metavar='int', default=6, type=int,
                   help='Radial order used for the SHORE fit. (Default: 6)')

    p.add_argument('--anisotropic_scaling', metavar='bool', default=True,
                   help='Anisotropique scaling.')

    p.add_argument('--pos_const', metavar='bool', default=True,
                   help='Positivity constraint.')

    p.add_argument('--lap_reg', metavar='int', default=1,
                   help='Laplacian regularization.')

    p.add_argument('--lap_weight', metavar='float', default=0.2,
                   help='Laplacian weighting in case of laplacian regularization.')

    p.add_argument('--sphere', default='repulsion724',
                   help='Type of sphere for the pdf compute.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load data, bvals, bvecs
    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.get_affine()
    data_shape = data.shape[:-1]

    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)
    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Load tractogram
    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()

    # Load label map
    vol_label = nib.load(args.in_label_map)
    label_map = vol_label.get_fdata()

    # Compute average directions for each voxel
    affine, data_shape, _, _ = sft.space_attributes
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape), args.sphere)
    todi_obj.compute_todi(sft.streamlines)
    avr_dir_todi = todi_obj.compute_average_dir()
    avr_dir_todi = todi_obj.reshape_to_3d(avr_dir_todi)

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

    r_sample = np.linspace(0.0, 0.025, args.nb_points)

    sections_len = np.floor_divide(np.max(label_map), args.nb_sections)

    for i in range(args.nb_sections):
        sec_vox = np.argwhere((label_map > sections_len * i) & (label_map >= sections_len * (i+1)))
        rand = np.random.randint(len(sec_vox), size=args.sample_size)
        vox_list = sec_vox[rand]
        pdf_sample = np.zeros((args.nb_sections, args.sample_size, args.nb_points))
        counter = 0
        for vox in vox_list:
            peak = avr_dir_todi[vox[0], vox[1], vox[2]]
            data_vox = data[vox[0], vox[1], vox[2]]
            mapmri_fit = mapmri_model.fit(data_vox)

            if np.max(np.abs(peak)) < 0.001:
                pdf_sample = np.delete(pdf_sample, counter, 0)

            else:
                r, theta, phi = cart2sphere(peak[0], peak[1], peak[2])
                theta = np.repeat(theta, args.nb_points)
                phi = np.repeat(phi, args.nb_points)

                x, y, z = sphere2cart(r_sample, theta, phi)

                r_points = np.vstack((x, y, z)).T

                pdf_sample[counter] = mapmri_fit.pdf(r_points)
                counter += 1
        np.savetxt(args.out_directory + str(i) + '_pdf_profile.csv', pdf_sample, fmt='%1.3f', delimiter=',')


if __name__ == "__main__":
    main()
