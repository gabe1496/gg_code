import numpy as np
from dipy.direction.peaks import (peak_directions,
                                  reshape_peaks_for_visualization)
from dipy.core.ndindex import ndindex
from dipy.core.geometry import sphere2cart, cart2sphere
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


def compute_peaks(glyph, sphere, npeaks=5):
    shape = glyph.shape[0:3]

    peaks_dirs = np.zeros((shape + (npeaks, 3)))
    for idx in ndindex(shape):

        direction, pk, ind = peak_directions(glyph[idx], sphere)
        n = min(npeaks, pk.shape[0])
        peaks_dirs[idx][:n] = direction[:n]
        peaks = reshape_peaks_for_visualization(peaks_dirs)

    return peaks


def segment_peaks_from_bundle(peaks, sft, mask, sphere, thr=45):
    affine, data_shape, _, _ = sft.space_attributes
    todi_obj = TrackOrientationDensityImaging(tuple(data_shape), sphere)
    todi_obj.compute_todi(sft.streamlines)
    avr_dir_todi = todi_obj.compute_average_dir()
    avr_dir_todi = todi_obj.reshape_to_3d(avr_dir_todi)

    ind_mask = np.argwhere(mask > 0)
    avr_dir_seg = avr_dir_todi[np.min(ind_mask[:, 0]):np.max(ind_mask[:, 0]) + 1,
                               np.min(ind_mask[:, 1]):np.max(ind_mask[:, 1]) + 1,
                               np.min(ind_mask[:, 2]):np.max(ind_mask[:, 2]) + 1]

    list_vox = np.indices((peaks.shape[0],
                           peaks.shape[1],
                           peaks.shape[2])).T.reshape(-1, 3)

    peaks_bundle = np.zeros_like(peaks)

    for ind in list_vox:
        peaks_vox = peaks[ind[0], ind[1], ind[2]]
        avg_dir = avr_dir_seg[ind[0], ind[1], ind[2]]
        peaks_vox = peaks_vox.reshape(5, 3)
        rep = np.dot(peaks_vox, avg_dir)
        rep2 = (rep / (np.linalg.norm(avg_dir) * np.linalg.norm(peaks_vox, axis=1)))
        angle = np.arccos(rep2)
        max_peak_ind = np.argsort(angle)[0]
        bundle_peak = np.zeros((15))
        if max_peak_ind <= np.deg2rad(thr):
            bundle_peak[0:3] = peaks_vox[max_peak_ind]
        peaks_bundle[ind[0], ind[1], ind[2]] = bundle_peak

    return avr_dir_seg, peaks_bundle

def compute_bundle_eap_profile_along_peaks(model, data, bundle_peaks, nb_points=20):
    list_vox = np.indices((bundle_peaks.shape[0],
                           bundle_peaks.shape[1],
                           bundle_peaks.shape[2])).T.reshape(-1, 3)

    r_sample = np.linspace(0.0, 0.025, nb_points)
    pdf_sample = np.zeros((list_vox.shape[0], nb_points))

    counter = 0
    for vox in list_vox:
        peak = bundle_peaks[vox[0], vox[1], vox[2]]
        data = data[vox[0], vox[1], vox[2]]
        mapmri_fit = model.fit(data)

        if np.max(np.abs(peak)) < 0.001:
            pdf_sample = np.delete(pdf_sample, counter, 0)

        else:
            r, theta, phi = cart2sphere(peak[0], peak[1], peak[2])
            theta = np.repeat(theta, nb_points)
            phi = np.repeat(phi, nb_points)

            x, y, z = sphere2cart(r_sample, theta, phi)

            r_points = np.vstack((x, y, z)).T

            pdf_sample[counter] = mapmri_fit.pdf(r_points)
            counter += 1

    return pdf_sample
