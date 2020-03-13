import numpy as np
import nibabel as nib
import sys
import gg_code.utils.process_data as gg_pd

def main():
    dmri_filename = sys.argv[1]
    dmri_bet_filename = sys.argv[2] #No extension
    output_filename = sys.argv[3]

    gg_pd.bet_brain(dmri_filename, dmri_bet_filename)

    vol = nib.load(dmri_filename)
    data = vol.get_fdata()
    affine = vol.get_affine()

    mask_load = nib.load(dmri_bet_filename + "_mask.nii.gz")
    mask = mask_load.get_fdata()

    data_seg = gg_pd.bet_diffusion_volume(data, mask)


    img = nib.Nifti2Image(data_seg, affine)
    nib.save(img, output_filename)


if __name__ == "__main__":
    main()