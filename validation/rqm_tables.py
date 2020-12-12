#!/usr/bin/env python
""" This script generate latex tables with statistics on RQM for
    each polarization and mode combination for the both platforms [S1A/S1B].

    run example:
            run rqm_tables.py input/npz/path output/path

    output:
            Text files with latex formatting of the results ('ALGORITHM_PLATFORM.txt')

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

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Generate latex tables from RQM data')
    #parser.add_argument('platform', choices=['S1A','S1B'])
    #parser.add_argument('mode', choices=['EW', 'IW'])
    #parser.add_argument('pol', choices=['VH', 'HV'])
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    #parser.add_argument('y_min')
    #parser.add_argument('y_max')
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def plot_results(d_plot, out_path):
    plt.clf()
    plt.rcParams['xtick.labelsize'] = 8
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color_list = ['#459EB0', '#B0459E', '#9EB045']
    gap = 0.25

    x = np.arange(len(d_plot.keys())+1)

    esa_data = []
    nersc_data = []
    diff_data = []

    for key in d_plot.keys():
        esa_data.append((d_plot[key]['Mean_ESA'], d_plot[key]['STD_ESA']))
        nersc_data.append((d_plot[key]['Mean_NERSC'], d_plot[key]['STD_NERSC']))
        diff_data.append((d_plot[key]['Mean_Diff'], d_plot[key]['STD_Diff']))

    # append last bar with the mean
    esa_m = np.nanmean(np.array(esa_data)[:, 0])
    esa_std = np.nanmean(np.array(esa_data)[:, 1])
    esa_data.append((esa_m, esa_std))

    nersc_m = np.nanmean(np.array(nersc_data)[:, 0])
    nersc_std = np.nanmean(np.array(nersc_data)[:, 1])
    nersc_data.append((nersc_m, nersc_std))

    diff_m = np.nanmean(np.array(diff_data)[:, 0])
    diff_std = np.nanmean(np.array(diff_data)[:, 1])
    diff_data.append((diff_m, diff_std))

    print(np.array(esa_data)[:,0])
    print(np.array(nersc_data)[:, 0])

    ax.bar(x, np.array(esa_data)[:,0],
           width=gap,
           color=color_list[0], yerr=np.array(esa_data)[:,1])

    ax.bar(x+gap, np.array(nersc_data)[:,0],
           width=gap,
           color=color_list[1], yerr=np.array(nersc_data)[:,1])

    ax.bar(x+gap*2, np.array(diff_data)[:,0],
           width=gap,
           color=color_list[2], yerr=np.array(diff_data)[:,1])

    ax.set_xticks(x+gap)
    labels = list(d_plot.keys())
    labels.append('Mean')
    ax.set_xticklabels(labels)

    ax.set_ylabel('RQM')
    ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.set_title('RQM: %s %s %s' % (args.platform, args.mode, args.pol))

    ax.legend(('ESA', 'NERSC', 'Diff.'))

    plt.savefig(out_path, bbox_inches='tight', dpi=300)

def get_mean_std(pref, data):
    res_ll = []
    for key in data.keys():
        if pref in key:
            res_ll.append(data[key])
            #print(data[key])
    return np.nanmean(np.concatenate(res_ll)), np.nanstd(np.concatenate(res_ll)), np.concatenate(res_ll)

def get_unique_regions(file_list):
    ''' Get unique combinations of mode, polarization and polarization mode '''
    ll = []
    for ifile in file_list:
        ll.append(os.path.basename(ifile).split('_')[-2])
    return list(set(ll))

def make_tbl(d_tbl, alg):
    for platform in platforms:
        print('\n%s\n' % platform)
        ll_name = 'rows_%s' % platform
        vars()[ll_name] = []
        vars()[ll_name].append(modes_ll)  # [ikey for ikey in d_plot[platform].keys()])

        for key_region in d_tbl[platform][modes_ll[2]].keys():

            try:
                iw_hv = '%.3f/%.3f' % (d_tbl[platform][modes_ll[1]][key_region]['Mean_%s' % alg],
                                       d_tbl[platform][modes_ll[1]][key_region]['STD_%s' % alg])
            except:
                iw_hv = '-'

            try:
                iw_vh = '%.3f/%.3f' % (d_tbl[platform][modes_ll[2]][key_region]['Mean_%s' % alg],
                                       d_tbl[platform][modes_ll[2]][key_region]['STD_%s' % alg])

            except:
                iw_vh = '-'

            try:
                ew_hv = '%.3f/%.3f' % (d_tbl[platform][modes_ll[3]][key_region]['Mean_%s' % alg],
                                       d_tbl[platform][modes_ll[3]][key_region]['STD_%s' % alg])
            except:
                ew_hv = '-'

            try:
                ew_vh = '%.3f/%.3f' % (d_tbl[platform][modes_ll[4]][key_region]['Mean_%s' % alg],
                                       d_tbl[platform][modes_ll[4]][key_region]['STD_%s' % alg])
            except:
                ew_vh = '-'

            vars()[ll_name].append([key_region,
                                    iw_vh, iw_hv, ew_hv, ew_vh])

            tbl = Texttable()
            tbl.set_cols_align(["c"] * 5)
            tbl.set_deco(Texttable.HEADER | Texttable.VLINES)
            tbl.set_cols_align(['l', 'c', 'c', 'c', 'c'])
            tbl.add_rows(vars()[ll_name])

            # Print table in readable form
        print(tbl.draw())

        tbl_body = tabulate(vars()[ll_name], headers='firstrow', tablefmt='latex')
        tbl_body = tbl_body.replace('{tabular}{lllll}', """{longtable}{lcccc} \\caption{RQM %s: %s}""" % (platform, alg)).replace('{tabular}','{longtable}')



        with open('%s/%s_%s_tbl.txt' % (args.out_path, alg, platform), 'w') as f:
            f.write(tbl_body)

pol_mode = {
    'VH': '1SDV',
    'HV': '1SDH',
}

args = parse_run_experiment_args()
os.makedirs(args.out_path, exist_ok=True)

platforms = ['S1A', 'S1B']

modes = [['IW','HV'],
         ['IW','VH'],
         ['EW','HV'],
         ['EW','VH']]

#regions = ['ANTARCTIC', 'ARCTIC', 'DESSERT', 'DOLLDRUMS', 'OCEAN']
d_plot = {}

for platform in platforms:
    d_plot[platform] = {}
    for imode in modes:
        imode_name =  '%s_%s' % (imode[0], imode[1])
        d_plot[platform][imode_name] = {}
        npz_list = glob.glob('%s/*%s*%s*%s*.npz' % (args.in_path, platform, imode[0], pol_mode[imode[1]]))
        #print(npz_list)

        # Get unique regions
        unq_file_masks = sorted(get_unique_regions(npz_list))

        for fmask in unq_file_masks:
            #print(fmask)
            npz_list = glob.glob('%s/*%s*%s*%s*%s*.npz' % (args.in_path, platform, imode[0], pol_mode[imode[1]], fmask))
            #print('Num of files: %s\n' % len(npz_list))

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

            d_plot[platform][imode_name][fmask] = {}
            d_plot[platform][imode_name][fmask]['Num_images'] = len(npz_list)
            d_plot[platform][imode_name][fmask]['Image_IDs'] = [os.path.basename(il).split('.')[0] for il in npz_list]

            # Print results
            m_esa, std_esa, data_esa = get_mean_std('ESA', res_d)
            d_plot[platform][imode_name][fmask]['Mean_ESA'] = m_esa
            d_plot[platform][imode_name][fmask]['STD_ESA'] = std_esa

            m_nersc, std_nersc, data_nersc = get_mean_std('NERSC', res_d)
            d_plot[platform][imode_name][fmask]['Mean_NERSC'] = m_nersc
            d_plot[platform][imode_name][fmask]['STD_NERSC'] = std_nersc

            diff = data_esa - data_nersc
            m_diff = np.nanmean(diff)
            std_diff = np.nanstd(diff)
            d_plot[platform][imode_name][fmask]['Mean_Diff'] = m_diff
            d_plot[platform][imode_name][fmask]['STD_Diff'] = std_diff

# Make tables from results
modes_ll = ['Validation site','IW_HV', 'IW_VH', 'EW_HV', 'EW_VH']

make_tbl(d_plot, 'ESA')
make_tbl(d_plot, 'NERSC')


