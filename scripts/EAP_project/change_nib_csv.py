#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_pdf_nib',
                   help='Path of the input file nibabel.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.in_pdf_nib, sep=',', header=None)
    data = df.values

    np.savetxt(args.in_pdf_nib[:-6]+'csv', data, fmt='%1.3f', delimiter=',')


if __name__ == "__main__":
    main()
