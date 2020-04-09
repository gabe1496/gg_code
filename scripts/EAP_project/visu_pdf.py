#! /usr/bin/env python
"""
Script to visualize pdf from EAP.
"""
import argparse

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.io.gradients import read_bvals_bvecs



def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_pdf',
                   help='Path of the input pdf.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()



if __name__ == "__main__":
    main()
