#! /usr/bin/env python


"""
Script to isolate a voxel from a diffusion image.
"""
import argparse
import nibabel as nib


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_diffusion',
                   help='Path of the input diffusion volume.')

    p.add_argument('x', type=int,
                   help='The first coordinate of the voxel.')

    p.add_argument('y', type=int,
                   help='The second coordinate of the voxel.')

    p.add_argument('z', type=int,
                   help='The thrid coordinate of the voxel.')

    p.add_argument('out_voxel',
                   help='Output filename of the isolated voxel.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    vol = nib.load(args.in_diffusion)
    data = vol.get_fdata()
    affine = vol.affine

    voxel = data[args.x, args.y, args.z]
    print(data.shape)
    print(voxel.shape)

    nib.save(nib.Nifti1Image(voxel, affine), args.out_voxel)


if __name__ == "__main__":
    main()
