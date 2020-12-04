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

    p.add_argument('in_pdf_cc',
                   help='Path of the input file csv for the CC.')

    p.add_argument('in_pdf_af',
                   help='Path of the input file csv for the AF.')

    p.add_argument('in_pdf_pt',
                   help='Path of the input file csv for the PYT.')

    p.add_argument('out_filename',
                   help='Path of the out graph.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df_cc = pd.read_csv(args.in_pdf_cc, sep=',', header=None)
    data_cc = df_cc.values
    df_af = pd.read_csv(args.in_pdf_af, sep=',', header=None)
    data_af = df_af.values
    df_pt = pd.read_csv(args.in_pdf_pt, sep=',', header=None)
    data_pt = df_pt.values

    x = np.linspace(0, 0.025, 20)

    mu_cc = np.mean(data_cc, axis=0)
    sigma_cc = data_cc.std(axis=0)
    mu_af = np.mean(data_af, axis=0)
    sigma_af = data_af.std(axis=0)
    mu_pt = np.mean(data_pt, axis=0)
    sigma_pt = data_pt.std(axis=0)

    fig, ax = plt.subplots(1)
    ax.plot(x, mu_cc, lw=2, label='CC', color='red')
    ax.plot(x, mu_af, lw=2, label='AF', color='green')
    ax.plot(x, mu_pt, lw=2, label='CST', color='blue')

    ax.fill_between(x, mu_cc+sigma_cc, mu_cc-sigma_cc, facecolor='red', alpha=0.5)
    ax.fill_between(x, mu_af+sigma_af, mu_af-sigma_af, facecolor='green', alpha=0.5)
    ax.fill_between(x, mu_pt+sigma_pt, mu_pt-sigma_pt, facecolor='blue', alpha=0.5)

    ax.set_title('EAP profile of different bundles depending on the radius')
    ax.legend(loc='upper right')
    ax.set_xlabel('Radius')
    ax.set_ylabel('EAP')
    ax.grid()
    plt.show()
    fig.savefig(args.out_filename)


if __name__ == "__main__":
    main()
