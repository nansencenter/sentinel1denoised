#!/usr/bin/env python
""" Range quality metric calculation and generate LATEX tables with results

    run example: run plot_qam.py polarization input_path result_path

    output:
            npz file with RQM for ESA and NERSC algorithms
            txt file with LATEX rable

"""
import argparse
import glob
import json
import os

import latextable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from texttable import Texttable

from s1denoise import Sentinel1Image

POL_MODES = {
    'HV': '1SDH',
    'VH': '1SDV',
}

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Range quality assessment from individual S1 ZIP or SAFE files')
    parser.add_argument('polarization', choices=['HV', 'VH'])
    parser.add_argument('s1_path')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def generate_latex_table(tab_out_fname, data_esa, data_nersc, print_RQM=True):
    """
    Compute average and std values of RQM, create latex table

    :param tab_out_fname: filename to output table
    :param data_esa: json filename with RQM for ESA data
    :param data_nersc: json filename with RQM for NERSC data
    :param print_RQM: print RQM in the LATEX table

    :return mean_esa: mean RQM for ESA
    :return mean_nersc: mean RQM for NERSC
    :return std_esa: std RQM for ESA
    :return std_nersc: std RQM for NERSC

    """
    with open(data_esa, 'rt') as esaf, open(data_nersc, 'rt') as nerscf:
        d_esa = json.load(esaf)
        d_nersc = json.load(nerscf)

    print('\n%s %s\n' % (d_esa, d_nersc))

    rows = []

    if print_RQM:
        rows.append(['No.', 'Scene ID', 'ESA', 'NERSC'])
        total_mean_esa = []
        total_mean_nersc = []
    else:
        rows.append(['No.', 'Scene ID'])

    for i, sar_id in enumerate(d_esa):
        mean_esa = d_esa[sar_id]['Mean'][0]
        mean_nersc = d_nersc[sar_id]['Mean'][0]

        if print_RQM:
            rows.append([i, sar_id, '%.3f' % mean_esa,
                         '%.3f' % mean_nersc])
            total_mean_esa.append(mean_esa)
            total_mean_nersc.append(mean_nersc)
        else:
            rows.append([i, sar_id])

    if print_RQM:
        rows.append(['Total', '', '%.3f' % np.nanmean(total_mean_esa),
                     '%.3f' % np.nanmean(total_mean_nersc)])
        rows.append(['STD', '', '%.3f' % np.nanstd(total_mean_esa),
                     '%.3f' % np.nanstd(total_mean_nersc)])

    table = Texttable()
    if print_RQM:
        table.set_cols_align(["c"] * 4)
    else:
        table.set_cols_align(["c"] * 2)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    #print('Tabulate Table:')
    #print(tabulate(rows, headers='firstrow'))

    #print('\nTexttable Table:')
    #print(table.draw())

    #print('\nTabulate Latex:')
    #print(tabulate(rows, headers='firstrow', tablefmt='latex'))

    # Save table to file
    with open(tab_out_fname, 'wt') as f:
        f.write(tabulate(rows, headers='firstrow', tablefmt='latex'))

    #print('\nTexttable Latex:')
    #print(latextable.draw_latex(table, caption="Caption."))

    return np.nanmean(total_mean_esa), np.nanmean(total_mean_nersc), np.nanstd(total_mean_esa), np.nanstd(total_mean_nersc)

def plot_results(png_out_fname, mean_esa, mean_nersc, std_esa, std_nersc):
    """
    Generate PNG file with bar / errorbar plot

    :param png_out_fname: filename to output table
    :param mean_esa: mean RQM for ESA
    :param mean_nersc: mean RQM for NERSC
    :param std_esa: std RQM for ESA
    :param std_nersc: std RQM for NERSC

    """
    materials = ['ESA', 'NERSC']
    plt.clf()
    title = '%s RQM mean/RQM std' % png_out_fname

    # Create lists for the plot
    x_pos = np.arange(len(materials))
    data = [mean_esa, mean_nersc]
    error = [std_esa, std_nersc]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, data, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('RQM')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Save the figure and close
    plt.tight_layout()
    plt.savefig(png_out_fname, bbox_inches='tight', dpi=300)
    plt.close()


args = parse_run_experiment_args()

out_prefix = os.path.join(args.out_dir, '%s' % os.path.basename(args.s1_path))
esa_out_fname = f'{out_prefix}_ESA_RQM.json'
nersc_out_fname = f'{out_prefix}_NERSC_RQM.json'
png_out_fname = f'{out_prefix}_NERSC_ESA_RQM.png'
tab_out_fname = f'{out_prefix}_table.png'

ffiles_mask = os.path.join(args.s1_path, '*%s*.zip' % POL_MODES[args.polarization])
ffiles = sorted(glob.glob(ffiles_mask))

# make directory for output npz files
os.makedirs(args.out_dir, exist_ok=True)

# Dictonary for ESA and NERSC QAM results
d_res_esa =  {}
d_res_nersc = {}

for fname in ffiles:
    id_name = os.path.basename(fname).split('.')[0]
    d_res_esa[id_name] = {}
    d_res_nersc[id_name] = {}

# Open first file to get keys of result
s1 = Sentinel1Image(ffiles[0])
res_qam = s1.qualityAssesment(args.polarization)
keys = list(res_qam['QAM_ESA'].keys())

for id_key in list(d_res_esa.keys()):
    for k in keys:
        d_res_esa[id_key][k] = []
        d_res_nersc[id_key][k] = []

for fname in ffiles:
    id_name = os.path.basename(fname).split('.')[0]
    s1 = Sentinel1Image(fname)
    res_qam = s1.qualityAssesment(args.polarization)
    keys = list(res_qam['QAM_ESA'].keys())
    for k in keys:
        d_res_esa[id_name][k].append(res_qam['QAM_ESA'][k])
        d_res_nersc[id_name][k].append(res_qam['QAM_NERSC'][k])


with open(esa_out_fname, 'wt') as fesa, open(nersc_out_fname, 'wt') as fnersc:
    json.dump(d_res_esa, fesa)
    json.dump(d_res_nersc, fnersc)

# Create LATEX table from obtained results
mean_esa, mean_nersc, std_esa, std_nersc = generate_latex_table(tab_out_fname, esa_out_fname, nersc_out_fname, print_RQM=True)

# Plot error bars
plot_results(png_out_fname, mean_esa, mean_nersc, std_esa, std_nersc)


