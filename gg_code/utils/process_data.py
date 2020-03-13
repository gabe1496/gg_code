import numpy as np
import nibabel as nib
import os


def crop_diffusion_volume(input_filename, dim_x, dim_y, dim_z, output_filename):
    vol = nib.load(input_filename)
    data = vol.get_fdata()
    affine = vol.get_affine()

    data = data[dim_x[0]:dim_x[1], dim_y[0]:dim_y[1], dim_z[0]:dim_z[1]]

    img = nib.Nifti2Image(data, affine)
    nib.save(img, output_filename)


def bet_brain(input_filename, output_filename, force=False):
    """
    Méthode qui appelle la fonction bet de fsl pour extraire le cerveau.
    """
    if force:
        os.system("bet " + input_filename + " " + output_filename + " -m -R -f")
    else:
        os.system("bet " + input_filename + " " + output_filename + " -m -R")


def bet_diffusion_volume(dmri_data, mask):
    """
    Méthode qui applique le masque du bet sur tout le volume 4D.
    """

    data_seg = np.zeros_like(dmri_data)
    for i in range(dmri_data.shape[3]):
        data_seg[:,:,:,i] = dmri_data[:,:,:,i] * mask
    
    return data_seg

