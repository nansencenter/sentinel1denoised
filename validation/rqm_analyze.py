#!/usr/bin/env python
""" Range quality metric averaging from json files

    run example: run rqm_analyze.py platform mode polarization input/path output/path

    output:
            png figure with mean values, STD and mean signed difference

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
    parser = argparse.ArgumentParser(description='Quality assessment aggregated statistics from individual npz files')
    parser.add_argument('platform', choices=['S1A', 'S1B'])
    parser.add_argument('mode', choices=['EW', 'IW'])
    parser.add_argument('polarization', choices=['HV', 'VH'])
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def plot_results(esa_files, nersc_files, mean_esa, mean_nersc, std_esa, std_nersc, mean_diff, mean_diff_std,
                 out_path, data_name):
    plt.clf()

    data_esa = []
    data_nersc = []
    data_diff = []

    labels = []

    ind = np.arange(len(esa_files)+2)  # the x locations for the groups
    width = 0.15  # the width of the bars

    for esa_file, nersc_file in zip(esa_files, nersc_files):
        with open(esa_file) as esa_json_file:
            d_esa = json.load(esa_json_file)

        with open(nersc_file) as nersc_json_file:
            d_nersc = json.load(nersc_json_file)

        for i, sar_id in enumerate(d_esa):
            mean_esa_region = d_esa[sar_id]['Mean'][0]
            data_esa.append(mean_esa_region)

        for i, sar_id in enumerate(d_nersc):
            mean_nersc_region = d_nersc[sar_id]['Mean'][0]
            data_nersc.append(mean_nersc_region)

        data_diff.append(mean_esa_region-mean_nersc_region)

        labels.append(os.path.basename(esa_file).split('_')[4])
    labels.append('Total')

    plot_mean_esa = np.nanmean(data_esa)
    plot_std_esa = np.nanstd(data_esa)
    plot_std_esa.append(std_esa)

    plot_mean_nersc = np.nanmean(data_nersc)
    plot_std_nersc = np.nanstd(data_nersc)
    plot_std_nersc.append(std_nersc)

    plot_mean_diff = np.nanmean(data_diff)
    plot_std_diff = np.nanstd(data_diff)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, plot_mean_esa, width, color='royalblue', yerr=plot_std_esa)
    rects2 = ax.bar(ind + width, plot_mean_nersc, width, color='seagreen', yerr=plot_std_nersc)
    rects3 = ax.bar(ind + width * 2, plot_mean_diff, width, color='red', yerr=plot_std_diff)

    # add some
    ax.set_ylabel('RQM')
    ax.set_title('%s %s %s' % (args.platform, args.mode, args.polarization))
    ax.set_xticks(ind + width)
    ax.set_xticklabels(labels)

    ax.legend((rects1[0], rects2[0], rects3[0]), ('ESA', 'NERSC', 'Diff.'))

    plt.savefig('%s/%s_RQM.png' % (out_path, data_name), bbox_inches='tight', dpi=300)

def get_aggregated_stat(esa_files, nersc_files):
    """ Get averaged data from many json files
    """
    total_mean_esa = []
    total_mean_nersc = []

    for esa_file, nersc_file in zip(esa_files, nersc_files):
        with open(esa_file) as esa_json_file:
            d_esa = json.load(esa_json_file)
        with open(nersc_file) as nersc_json_file:
            d_nersc = json.load(nersc_json_file)

        for i, sar_id in enumerate(d_esa):
            mean_esa = d_esa[sar_id]['Mean'][0]
            total_mean_esa.append(mean_esa)

        for i, sar_id in enumerate(d_nersc):
            mean_nersc = d_nersc[sar_id]['Mean'][0]
            total_mean_nersc.append(mean_nersc)

    # Calculate mean difference
    mean_diff = np.array(total_mean_esa) - np.array(total_mean_nersc)

    return np.nanmean(total_mean_esa), np.nanmean(total_mean_nersc),\
           np.nanstd(total_mean_esa), np.nanstd(total_mean_nersc), np.nanmean(mean_diff), np.nanstd(mean_diff)


pol_mode = {
    'VH': '1SDV',
    'HV': '1SDH',
}

args = parse_run_experiment_args()
os.makedirs(args.out_path, exist_ok=True)
npz_list = glob.glob('%s/*%s*%s*%s*.npz' % (args.in_path, args.platform, args.mode, pol_mode[args.polarization]))

total_esa_data = []
total_nersc_data = []
total_diff_data = []

# Create lists based on keys names
a = npz_list[0]
a = np.load(a)
d_npz = dict(zip((k for k in a), (a[k] for k in a)))
for key in d_npz.keys():
    var_name = '%s_ll' % key
    vars()[var_name] = []

# Collect data into vars
for a in npz_list:
    a = np.load(a)
    d_npz = dict(zip((k for k in a), (a[k] for k in a)))

    for key in d_npz.keys():
        var_name = '%s_ll' % key
        # Add mean value for subswath margin for each scene
        vars()[var_name].append(np.nanmean(d_npz[key]))




'''
# Get aggregated stat
mean_esa, mean_nersc, std_esa, std_nersc, \
        mean_diff, mean_diff_std = \
    get_aggregated_stat(esa_json_list, nersc_json_list)

plot_results(esa_json_list, nersc_json_list, mean_esa, mean_nersc, std_esa, std_nersc,
                     mean_diff, mean_diff_std, args.out_dir, data_name)
                     
'''

