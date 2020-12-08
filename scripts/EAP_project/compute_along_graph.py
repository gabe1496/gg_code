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

    p.add_argument('in_pdf',
                   help='Path of the input files.')

    p.add_argument('out_filename',
                   help='Path of the out graph.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # mu = np.zeros(5)
    # sigma =np.zeros(5)
    # list_colors = ['lightgreen', 'lime', 'limegreen', 'green', 'darkgreen']
    # list_colors = ['lightsteelblue', 'cornflowerblue', 'royalblue', 'blue', 'navy']
    list_colors = ['mistyrose', 'tomato', 'red', 'firebrick', 'darkred']
    x = np.linspace(0, 0.025, 20)

    fig, ax = plt.subplots(1)

    for i in range(5):
        df = pd.read_csv(args.in_pdf+ str(i)+'.csv', sep=',', header=None)
        data = df.values
        mu = np.mean(data, axis=0)
        sigma= data.std(axis=0)
        ax.plot(x, mu, lw=2, label=str(i), color=list_colors[i])
        # ax.fill_between(x, mu+sigma, mu-sigma, facecolor=list_colors[i], alpha=0.3)

    
    # ax.plot(x, mu_cc, lw=2, label='CC', color='red')
    # ax.plot(x, mu_af, lw=2, label='AF', color='green')
    # ax.plot(x, mu_pt, lw=2, label='CST', color='blue')

    # ax.fill_between(x, mu_cc+sigma_cc, mu_cc-sigma_cc, facecolor='red', alpha=0.5)
    # ax.fill_between(x, mu_af+sigma_af, mu_af-sigma_af, facecolor='green', alpha=0.5)
    # ax.fill_between(x, mu_pt+sigma_pt, mu_pt-sigma_pt, facecolor='blue', alpha=0.5)

    ax.set_title('EAP profile depending on the radius')
    ax.legend(loc='upper right')
    ax.set_xlabel('Radius')
    ax.set_ylabel('EAP')
    ax.grid()
    plt.show()
    fig.savefig(args.out_filename)


if __name__ == "__main__":
    main()
