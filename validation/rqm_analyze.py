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

def plot_results(d_plot, out_path):
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = []
    data.append(d_plot['Mean_ESA'])
    data.append(d_plot['Mean_NERSC'])
    data.append(d_plot['Mean_Diff'])
    print(data)

    color_list = ['g', 'b', 'r']
    gap = .8 / len(data)
    for i, row in enumerate(data):
        try:
            X = np.arange(len(row))
        except:
            X = np.arange(1)
        plt.bar(X + i * gap, row,
                width=gap,
                color=color_list[i % len(color_list)])

    ax.set_ylabel('RQM')
    ax.set_title('%s %s %s' % (args.platform, args.mode, args.polarization))
    #ax.set_xticks(ind + width)
    #labels = d_plot.keys()
    #ax.set_xticklabels(labels)
    ax.legend((data[0], data[1], data[2]), ('ESA', 'NERSC', 'Diff.'))

    plt.savefig(out_path, bbox_inches='tight', dpi=300)

def get_mean_std(pref, data):
    res_ll = []
    for key in data.keys():
        if pref in key:
            res_ll.append(data[key])
    return np.nanmean(np.concatenate(res_ll)), np.nanstd(np.concatenate(res_ll)), np.concatenate(res_ll)

pol_mode = {
    'VH': '1SDV',
    'HV': '1SDH',
}

args = parse_run_experiment_args()
os.makedirs(args.out_path, exist_ok=True)
npz_list = glob.glob('%s/*%s*%s*%s*.npz' %
                     (args.in_path, args.platform, args.mode, pol_mode[args.polarization]))

total_esa_data = []
total_nersc_data = []
total_diff_data = []

res_d = {}

# Create lists based on keys names
a = npz_list[0]
a = np.load(a)
d_npz = dict(zip((k for k in a), (a[k] for k in a)))

for key in d_npz.keys():
    res_d['%s_ll' % key] = []

# Collect data for each margin
for key in d_npz.keys():
    for a in npz_list:
        var_name = '%s_ll' % key
        res_d[var_name].append(d_npz[key])
    arr = np.concatenate(res_d[var_name])
    res_d[var_name] = arr

d_plot = {}

# Print results
m_esa, std_esa, data_esa = get_mean_std('ESA', res_d)
d_plot['Mean_ESA'] = m_esa
d_plot['STD_ESA'] = std_esa
print('\n#####\nESA mean/STD: %.3f/%.3f\n#####\n' % (m_esa, std_esa))

m_nersc, std_nersc, data_nersc = get_mean_std('NERSC', res_d)
d_plot['Mean_NERSC'] = m_nersc
d_plot['STD_NERSC'] = std_nersc
print('\n#####\nNERSC mean/STD: %.3f/%.3f\n#####\n' % (m_nersc, std_nersc))

diff = data_esa - data_nersc
m_diff = np.nanmean(diff)
std_diff = np.nanstd(diff)
d_plot['Mean_Diff'] = m_diff
d_plot['STD_Diff'] = std_diff
print('\n#####\nDifference mean/STD: %.3f/%.3f\n#####\n' % (m_diff, std_diff))

plot_results(d_plot, 'test_plot.png')
