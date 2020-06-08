#! /usr/bin/env python
import nibabel as nib
import sys
import gg_code.utils.process_data as gg_pd


def main():
    anat_filename = sys.argv[1]
    bet_mask = sys.argv[2]
    output_filename = sys.argv[3]

    vol = nib.load(anat_filename)
    data = vol.get_fdata()
    affine = vol.get_affine()

    mask_load = nib.load(bet_mask)
    mask = mask_load.get_fdata()

    data_seg = data * mask

    img = nib.Nifti1Image(data_seg, affine)
    nib.save(img, output_filename)


if __name__ == "__main__":
    main()
