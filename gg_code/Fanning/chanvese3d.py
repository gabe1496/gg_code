# ------------------------------------------------------------------------
# Region Based Active Contour Segmentation
#
# seg = chanvese3d(I,init_mask,max_its,alpha,display)
#
# Inputs: img         3D image
#         init_mask   Initialization (1 = foreground, 0 = background)
#         max_its     Number of iterations to run segmentation for
#         alpha       (optional) Weight of smoothing term
#                       higher = smoother.  default = 0.2
#         display     (optional) displays intermediate outputs
#                       default = true
#
# Outputs: seg        Final segmentation mask (1 = foreground, 0 = background)
#
# Description: This code implements the paper: "Active Contours Without
# Edges" By Chan Vese. This is a nice way to segment images whose
# foregrounds and backgrounds are statistically different and homogeneous.
#
# Coded by: Shawn Lankton
# Modified by Etienne St-Onge
# ------------------------------------------------------------------------
import os
import nibabel
import numpy as np
from numpy.linalg import norm
import scipy.ndimage as nd
import matplotlib.pyplot as plt
# import time

from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh
import scilpy.tractanalysis.todi_util as todi_u

import nibabel as nib

eps = np.sqrt(np.finfo(float).eps)


def chanvese3d(img, init_mask, max_its=20, alpha=0.2, thresh=0,
               color='r', display=False):
    # data_path = os.path.dirname(os.path.abspath(__file__)) + "/../mask/"
    # ds = 1
    # nib_file = nibabel.load(data_path + "2401_CST_L_binary_mask.nii.gz")
    # img = nib_file.get_data()[::ds, ::ds, ::ds]
    # mask = nibabel.load(data_path + "2401_CST_L_binary_mask.nii.gz").get_data()[::ds, ::ds, ::ds]

    # Create a signed distance map (SDF) from mask
    img = img.astype(np.float32)
    print img
    phi = mask2phi(init_mask)
    print phi.shape

    # Init
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0

    while (its <= max_its and not stop):

        # get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) > 0:
            # Intermediate output
            if display:
                # if np.mod(its, 10) == 0:
                print 'iteration:', its
                showCurveAndPhi(img, phi, color, its)

            else:
                if np.mod(its, 10) == 0:
                    print 'iteration:', its

            # Interior / Exterior mean
            in_mean = np.mean(img[phi <= 0.0])
            out_mean = np.mean(img[phi > 0.0])

            # Force from image information
            # distance to mean value
            F = (img.flat[idx] - in_mean)**2 - (img.flat[idx] - out_mean)**2
            # Curvature penalty
            curvature = get_curvature(phi, idx)
            #curvature[curvature < 0.0] = 0.0
            #curvature **= 2

            # Step / gradient descent to minimize energy
            dphidt = F / np.max(np.abs(F)) + alpha * curvature
            #dphidt = alpha * curvature

            # CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)

            # Evolve the curve
            phi.flat[idx] += dt * dphidt

            # Keep SDF smooth (re-initialize)
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

        else:
            break

    # Final output
    if display:
        showCurveAndPhi(img, phi, color, its)
        print("DONE")
        plt.show()

    # Make mask from SDF
    seg = phi <= 0



    # mask = nibabel.load(data_path + "2401_CST_L_binary_mask.nii.gz").get_data()[::ds, ::ds, ::ds]
    # seg, phi, its = chanvese3d(img, mask, max_its=4, alpha=1, display=False)
    # img = nib.Nifti1Image(phi, affine)
    # img.to_filename(data_path + "phi.nii.gz")
    # img = nib.Nifti1Image(seg.astype(np.uint16), affine)
    # img.to_filename(data_path + "seg.nii.gz")

    return phi


def flow3d(img, init_mask, nb_its=1, alpha=0.2,
           sphere_type='repulsion100', sh_order=4):
    # Create a signed distance map (SDF) from mask
    img = img.astype(np.float32)
    phi = mask2phi(init_mask)

    max_v = 1.2
    slow_v = 10.0

    # Todi init ( care memory size )
    sphere = get_sphere(sphere_type)
    nb_sphere_vts = len(sphere.vertices)
    nb_voxel = np.prod(img.shape)
    todi = np.zeros([nb_voxel, nb_sphere_vts], dtype=np.float)

    # Init
    its = 0
    while (its < nb_its):
        print its
        #showCurveAndPhi(img, phi, 'r')

        # get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= max_v, phi >= -max_v))

        if len(idx) > 0:
            # Interior / Exterior mean
            # in_mean = np.mean(img[phi <= 0.0])
            # out_mean = np.mean(img[phi > 0.0])

            # Force from image information
            # distance to mean value
            # F = (img.flat[idx] - in_mean)**2 - (img.flat[idx] - out_mean)**2

            # Curvature penalty
            curvature = get_curvature(phi, idx)
            curvature[curvature < 0.0] = 0.0
            #curvature **= 2

            # Step / gradient descent to minimize energy
            # dphidt = F / np.max(np.abs(F)) + alpha * curvature
            dphidt = alpha * curvature

            # CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)
            # dt = (0.45 / (np.max(np.abs(dphidt)) + eps)) / slow_v

            grad_of_phi = grad_phi(phi)
            grad_norm_of_phi = np.sqrt(np.sum(np.square(grad_of_phi), axis=0))

            # TODI computation
            # Compute local direction (gradient) on the curve
            grad_of_idx = np.zeros([len(idx), 3])
            grad_of_idx[:, 0] = grad_of_phi[0].flat[idx]
            grad_of_idx[:, 1] = grad_of_phi[1].flat[idx]
            grad_of_idx[:, 2] = grad_of_phi[2].flat[idx]

            grad_norm = grad_norm_of_phi.flat[idx]
            mask = np.logical_and(grad_norm > eps, curvature > eps)
            n_idx = idx[mask]

            dir_of_idx = todi_u.normalize_vectors(grad_of_idx[np.where(mask)])

            # Estimate each direction on a discrete sphere
            sph_id_of_idx = todi_u.get_dir_to_sphere_id(dir_of_idx, sphere.vertices)

            # Values for on the curve (close to 0)
            # weights = max_v - np.abs(phi.flat[n_idx])

            # Generate mask from streamlines points
            todi[n_idx, sph_id_of_idx] += grad_norm[mask]  # weights

            # UPDATE THE CURVE
            # Evolve the curve
            phi.flat[idx] += dt * dphidt

            # Keep SDF smooth (re-initialize)
            phi = phi - 0.5 * np.sign(phi) * (grad_norm_of_phi - 1.0)

        its += 1

    todi = todi_u.normalize_vectors(todi)
    sh_todi = sf_to_sh(todi, sphere, sh_order, "mrtrix", 0.006)
    return np.reshape(sh_todi, [img.shape[0], img.shape[1], img.shape[2], -1])


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# -- AUXILIARY FUNCTIONS ----------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def bwdist(a):
    """
    this is an intermediary function, 'a' has only True, False vals,
    so we convert them into 0, 1 values -- in reverse. True is 0,
    False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)


# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + init_a - 0.5
    return phi


# Displays the image with curve superimposed
def showCurveAndPhi(img, phi, color, its):
    # subplot(numRows, numCols, plotNum)

    plt.gcf().clear()
    plt.subplot(321)
    plt.imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    plt.contour(phi[:, :, img.shape[2] // 2], 0, colors=color)

    plt.subplot(322)
    plt.imshow(phi[:, :, img.shape[2] // 2])
    plt.colorbar()

    plt.subplot(323)
    plt.imshow(img[:, img.shape[1] // 2, :], cmap='gray')
    plt.contour(phi[:, img.shape[1] // 2, :], 0, colors=color)

    plt.subplot(324)
    plt.imshow(phi[:, img.shape[1] // 2, :])

    plt.subplot(325)
    plt.imshow(img[img.shape[0] // 2, :, :], cmap='gray')
    plt.contour(phi[img.shape[0] // 2, :, :], 0, colors=color)

    plt.subplot(326)
    plt.imshow(phi[img.shape[0] // 2, :, :])
    # plt.savefig("/home/local/USHERBROOKE/greg2707/Recherche/gg_code/scripts/Fanning/Data/phi_test"+str(its)+".png")
    # plt.pause(1)
    #plt.draw()
    #plt.show(block=True)
    # time.sleep(1)


# Compute curvature along SDF
def get_curvature(phi, idx):
    dimz, dimy, dimx = phi.shape
    zyx = np.array([np.unravel_index(i, phi.shape) for i in idx])
    x = zyx[:, 2]
    y = zyx[:, 1]
    z = zyx[:, 0]

    # Get subscripts of neighbors
    xm1 = x - 1
    ym1 = y - 1
    zm1 = z - 1
    xp1 = x + 1
    yp1 = y + 1
    zp1 = z + 1

    # Bounds checking
    xm1[xm1 < 0] = 0
    ym1[ym1 < 0] = 0
    zm1[zm1 < 0] = 0

    xp1[xp1 >= dimx] = dimx - 1
    yp1[yp1 >= dimy] = dimy - 1
    zp1[zp1 >= dimz] = dimz - 1

    # Get central derivatives of SDF at x,y
    dx = (phi[z, y, xm1] - phi[z, y, xp1]) / 2.0  # (l-r)/2
    dxx = phi[z, y, xm1] - 2.0 * phi[z, y, x] + phi[z, y, xp1]  # l-2c+r
    dx2 = dx * dx

    dy = (phi[z, ym1, x] - phi[z, yp1, x]) / 2.0  # (u-d)/2
    dyy = phi[z, ym1, x] - 2.0 * phi[z, y, x] + phi[z, yp1, x]  # u-2c+d
    dy2 = dy * dy

    dz = (phi[zm1, y, x] - phi[zp1, y, x]) / 2.0  # (b-f)/2
    dzz = phi[zm1, y, x] - 2.0 * phi[z, y, x] + phi[zp1, y, x]  # b-2c+f
    dz2 = dz * dz

    # (ul+dr-ur-dl)/4
    dxy = (phi[z, ym1, xm1] + phi[z, yp1, xp1]
           - phi[z, ym1, xp1] - phi[z, yp1, xm1]) / 4.0

    # (lf+rb-rf-lb)/4
    dxz = (phi[zp1, y, xm1] + phi[zm1, y, xp1]
           - phi[zp1, y, xp1] - phi[zm1, y, xm1]) / 4.0

    # (uf+db-df-ub)/4
    dyz = (phi[zp1, ym1, x] + phi[zm1, yp1, x]
           - phi[zp1, yp1, x] - phi[zm1, ym1, x]) / 4.0

    # Compute curvature (Kappa)
    curvature = ((dxx * (dy2 + dz2) +
                  dyy * (dx2 + dz2) +
                  dzz * (dx2 + dy2) -
                  2.0 * (dx * dy * dxy + dx * dz * dxz + dy * dz * dyz)
                  ) / (dx2 + dy2 + dz2 + eps))
    return curvature


def mymax(a, b):
    # return (a + b + np.abs(a - b)) / 2
    return np.maximum(a, b)


def grad_phi(phi):
    g_phi = np.zeros([3, phi.shape[0], phi.shape[1], phi.shape[2]])
    phi_pos_ind = (phi > 0.0)
    phi_neg_ind = (phi < 0.0)

    dx_f = np.zeros(phi.shape)
    dy_f = np.zeros(phi.shape)
    dz_f = np.zeros(phi.shape)
    dx_f[0:-1] = np.diff(phi, axis=0)
    dy_f[:, 0:-1] = np.diff(phi, axis=1)
    dz_f[:, :, 0:-1] = np.diff(phi, axis=2)

    dx_b = np.roll(dx_f, 1, axis=0)
    dy_b = np.roll(dy_f, 1, axis=1)
    dz_b = np.roll(dz_f, 1, axis=2)

    g_phi[0][phi_pos_ind] = mymax(dx_b[phi_pos_ind].clip(min=0.0),
                                  -dx_f[phi_pos_ind].clip(max=0.0))
    g_phi[1][phi_pos_ind] = mymax(dy_b[phi_pos_ind].clip(min=0.0),
                                  -dy_f[phi_pos_ind].clip(max=0.0))
    g_phi[2][phi_pos_ind] = mymax(dz_b[phi_pos_ind].clip(min=0.0),
                                  -dz_f[phi_pos_ind].clip(max=0.0))

    g_phi[0][phi_neg_ind] = mymax(-dx_b[phi_neg_ind].clip(max=0.0),
                                  dx_f[phi_neg_ind].clip(min=0.0))
    g_phi[1][phi_neg_ind] = mymax(-dy_b[phi_neg_ind].clip(max=0.0),
                                  dy_f[phi_neg_ind].clip(min=0.0))
    g_phi[2][phi_neg_ind] = mymax(-dz_b[phi_neg_ind].clip(max=0.0),
                                  dz_f[phi_neg_ind].clip(min=0.0))
    return g_phi


def grad_norm(phi):
    return np.sqrt(np.sum(np.square(grad_phi(phi)), axis=0))


def sussman(phi, dt):
    return phi - dt * np.sign(phi) * (grad_norm(phi) - 1.0)


# Roll matrix derivatives
def forward_diff(img, axis=0):
    return np.roll(img, -1, axis=axis) - img


def backward_diff(img, axis=0):
    return img - np.roll(img, 1, axis=axis)


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    diff = p_mask.astype(np.float) - n_mask.astype(np.float)
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
    return c




