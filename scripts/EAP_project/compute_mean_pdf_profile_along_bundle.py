#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np
from tqdm import tqdm

from dipy.reconst.mapmri import MapmriModel
from dipy.core.gradients import gradient_table
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractanalysis.grid_intersections import grid_intersections
from scilpy.utils.bvec_bval_tools import check_b0_threshold


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Path of the bundle file.')

    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('bvals',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('bvecs',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('weight_map',
                   help='Path of the output weight map.')

    p.add_argument('eap_mean_map',
                   help='Path of the output mean eap map.')

    p.add_argument('--length_weighting', action='store_true',
                   help='If set, will weigh the EAP values according to '
                        'segment lengths. [%(default)s]')

    p.add_argument('--nb_points', metavar='int', default=20,
                   help='Number of points to sample along the peaks.')

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

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

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
    all_crossed_indices = grid_intersections(sft.streamlines)

    weight_map = np.zeros(shape=data_shape)
    eap_map_shape = list(data_shape)
    eap_map_shape.append(args.nb_points)
    eap_map = np.zeros(shape=eap_map_shape)

    crossed_indices = all_crossed_indices[20]
    segments = crossed_indices[1:] - crossed_indices[:-1]
    seg_lengths = np.linalg.norm(segments, axis=1)
    print('segments', segments.shape)

    # Remove points where the segment is zero.
    # This removes numpy warnings of division by zero.
    non_zero_lengths = np.nonzero(seg_lengths)[0]
    segments = segments[non_zero_lengths]
    seg_lengths = seg_lengths[non_zero_lengths]
    print('segments non zero', segments.shape)

    # Those starting points are used for the segment vox_idx computations
    strl_start = crossed_indices[non_zero_lengths]
    vox_indices = (strl_start + (0.5 * segments)).astype(int)
    print('vox indices', vox_indices.shape)

    normalization_weights = np.ones_like(seg_lengths)
    if args.length_weighting:
        normalization_weights = seg_lengths / np.linalg.norm(vol.header.get_zooms()[:3])

    for vox_idx, seg, norm_weight in tqdm(zip(vox_indices,
                                              segments,
                                              normalization_weights),
                                          total=vox_indices.shape[0]):
        vox_idx = tuple(vox_idx)
        data_vox = data[vox_idx]
        mapmri_fit = mapmri_model.fit(data_vox)

        r, theta, phi = cart2sphere(seg[0], seg[1], seg[2])
        theta = np.repeat(theta, args.nb_points)
        phi = np.repeat(phi, args.nb_points)
        x, y, z = sphere2cart(r_sample, theta, phi)
        r_points = np.vstack((x, y, z)).T

        pdf = mapmri_fit.pdf(r_points) * norm_weight

        weight_map[vox_idx] += norm_weight
        eap_map[vox_idx] += pdf

    nib.save(nib.Nifti1Image(weight_map, affine), args.weight_map)
    nib.save(nib.Nifti1Image(eap_map, affine), args.eap_mean_map)


if __name__ == "__main__":
    main()


# r_sample = np.linspace(0.0, 0.025, args.nb_points)
#     all_crossed_indices = grid_intersections(sft.streamlines)

#     weight_map = np.zeros(shape=data_shape)
#     eap_map_shape = list(data_shape)
#     eap_map_shape.append(args.nb_points)
#     eap_map = np.zeros(shape=eap_map_shape)

#     for crossed_indices in all_crossed_indices:
#         segments = crossed_indices[1:] - crossed_indices[:-1]
#         seg_lengths = np.linalg.norm(segments, axis=1)

#         # Remove points where the segment is zero.
#         # This removes numpy warnings of division by zero.
#         non_zero_lengths = np.nonzero(seg_lengths)[0]
#         segments = segments[non_zero_lengths]
#         seg_lengths = seg_lengths[non_zero_lengths]

#         # Those starting points are used for the segment vox_idx computations
#         strl_start = crossed_indices[non_zero_lengths]
#         vox_indices = (strl_start + (0.5 * segments)).astype(int)

#         normalization_weights = np.ones_like(seg_lengths)
#         if args.length_weighting:
#             normalization_weights = seg_lengths / np.linalg.norm(vol.header.get_zooms()[:3])

#         for vox_idx, seg, norm_weight in zip(vox_indices,
#                                              segments,
#                                              normalization_weights):
#             vox_idx = tuple(vox_idx)
#             data_vox = data[vox_idx]
#             mapmri_fit = mapmri_model.fit(data_vox)

#             r, theta, phi = cart2sphere(seg[0], seg[1], seg[2])
#             theta = np.repeat(theta, args.nb_points)
#             phi = np.repeat(phi, args.nb_points)
#             x, y, z = sphere2cart(r_sample, theta, phi)
#             r_points = np.vstack((x, y, z)).T

#             pdf = mapmri_fit.pdf(r_points) * norm_weight

#             weight_map[vox_idx] += norm_weight
#             eap_map[vox_idx] += pdf

#     nib.save(nib.Nifti1Image(weight_map, affine), args.weight_map)
#     nib.save(nib.Nifti1Image(eap_map, affine), args.eap_mean_map)