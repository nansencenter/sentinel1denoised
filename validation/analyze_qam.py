#!/usr/bin/env python
""" Range quality metric analysis to plot statistics and generate Latex tables

    run example: run analyze_qam.py input_path result_path

    output:
            png figure with mean values, std and mean signed difference
            txt file with LATEX table

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
import json

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Range quality assessment aggregated statistics from json files')
    parser.add_argument('json_path')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def generate_latex_table(data_esa, data_nersc, out_path, data_name, print_RQM = True):
    """
    :param data_esa: json file with RQM for ESA data
    :param data_nersc: json file with RQM for NERSC data
    :param print_RQM: print RQM in the LATEX table
    :param out_path: output folder
    :param data_name: base name for output files
    :param print_RQM statisctics in Latex table (additional table will be created)

    :return: LATEX table in txt format
             PNG file with error bars
    """

    with open(data_esa) as esa_json_file:
        d_esa = json.load(esa_json_file)

    with open(data_nersc) as nersc_json_file:
        d_nersc = json.load(nersc_json_file)

    print('\n%s %s\n' % (d_esa, d_nersc))

    rows = []
    rows.append(['No.', 'Scene ID'])

    if print_RQM:
        rows_RQM = []
        rows_RQM.append(['No.', 'Scene ID', 'ESA', 'NERSC'])
        total_mean_esa = []
        total_mean_nersc = []
    else:
        rows.append(['No.', 'Scene ID'])

    for i, sar_id in enumerate(d_esa):
        mean_esa = d_esa[sar_id]['Mean'][0]
        mean_nersc = d_nersc[sar_id]['Mean'][0]
        total_mean_esa.append(mean_esa)
        total_mean_nersc.append(mean_nersc)

        if print_RQM:
            rows_RQM.append([i, sar_id, '%.3f' % mean_esa,
                         '%.3f' % mean_nersc])

        rows.append([i, sar_id])

    if print_RQM:
        rows_RQM.append(['Total', '', '%.3f' % np.nanmean(total_mean_esa),
                     '%.3f' % np.nanmean(total_mean_nersc)])
        rows_RQM.append(['STD', '', '%.3f' % np.nanstd(total_mean_esa),
                     '%.3f' % np.nanstd(total_mean_nersc)])

    if print_RQM:
        table_RQM = Texttable()
        table_RQM.set_cols_align(["c"] * 4)
        table_RQM.set_deco(Texttable.HEADER | Texttable.VLINES)
        table_RQM.add_rows(rows_RQM)

    table = Texttable()
    table.set_cols_align(["c"] * 2)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)

    # Save table to file
    if print_RQM:
        with open('%s/%s_table_RQM.txt' % (out_path, data_name), 'w') as f:
            f.write(tabulate(rows_RQM, headers='firstrow', tablefmt='latex'))

    with open('%s/%s_table.txt' % (out_path, data_name), 'w') as f:
        f.write(tabulate(rows, headers='firstrow', tablefmt='latex'))

    # Calculate mean difference
    mean_diff = np.array(total_mean_esa) - np.array(total_mean_nersc)

    return np.nanmean(total_mean_esa), np.nanmean(total_mean_nersc),\
           np.nanstd(total_mean_esa), np.nanstd(total_mean_nersc), np.nanmean(mean_diff), np.nanstd(mean_diff)

def plot_results(mean_esa, mean_nersc, std_esa, std_nersc, mean_diff, mean_diff_std, out_path, data_name):
    labels = ['ESA', 'NERSC', 'Mean difference\n(ESA-NERSC)']
    plt.clf()
    title = '%s RQM' % data_name

    # Create lists for the plot
    x_pos = np.arange(len(labels))
    data = [mean_esa, mean_nersc, mean_diff]
    error = [std_esa, std_nersc, mean_diff_std]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, data, yerr=error, align='center', alpha=0.5,
           color='#82B446', ecolor='black', capsize=10)
    ax.set_ylabel('RQM')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('%s/%s_RQM.png' % (out_path, data_name), bbox_inches='tight', dpi=300)

def compute_mean_rqm(data_esa, data_nersc):
    """Function to compute mean values for RQM
    :param data_esa: json file with RQM for ESA data
    :param data_nersc: json file with RQM for NERSC data
    :return: mean and STD values for ESA, NERSC data and their difference
    """

    total_mean_esa = []
    total_mean_nersc = []

    with open(data_esa) as esa_json_file:
        d_esa = json.load(esa_json_file)

    with open(data_nersc) as nersc_json_file:
        d_nersc = json.load(nersc_json_file)

    for i, sar_id in enumerate(d_esa):
        mean_esa = d_esa[sar_id]['Mean'][0]
        mean_nersc = d_nersc[sar_id]['Mean'][0]
        total_mean_esa.append(mean_esa)
        total_mean_nersc.append(mean_nersc)

    esa_mean = np.nanmean(total_mean_esa)
    nersc_mean = np.nanmean(total_mean_nersc)
    esa_std = np.nanstd(total_mean_esa)
    nersc_std = np.nanstd(total_mean_nersc)

    # Calculate mean difference
    diff = np.array(total_mean_esa) - np.array(total_mean_nersc)
    diff_mean = np.nanmean(diff)
    diff_std_ = np.nanstd(diff)

    return esa_mean, esa_std, nersc_mean, nersc_std, diff_mean, diff_std_

args = parse_run_experiment_args()

ffiles_mask_esa = os.path.join(args.json_path, '*ESA*RQM*.json')
ffiles_esa = sorted(glob.glob(ffiles_mask_esa))

ffiles_mask_nersc = os.path.join(args.json_path, '*NERSC*RQM*.json')
ffiles_nersc = sorted(glob.glob(ffiles_mask_nersc))

for esa_out_fname in ffiles_esa:
    for nersc_out_fname in ffiles_nersc:

        # Create LATEX tables from results
        data_name = os.path.basename(esa_out_fname).replace('_ESA','').split('.')[0]
        mean_esa, mean_nersc, std_esa, std_nersc, \
        mean_diff, mean_diff_std = \
            generate_latex_table(esa_out_fname, nersc_out_fname, args.out_dir, data_name, print_RQM=True)

        # Plot error bars
        plot_results(mean_esa, mean_nersc, std_esa, std_nersc,
                     mean_diff, mean_diff_std, args.out_dir, data_name)