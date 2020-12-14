#!/usr/bin/env python
""" Range quality metric plotting and averaging for each sensing mode
    for both platforms [S1A/S1B] from individual npz files

    run example:
            run rqm_plot.py input/npz/path output/path

    output:
            PNG figures with RQM mean values, STD and mean signed difference between the algorithms

"""

import argparse
import glob
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Plotting aggregated statistics on RQM from individual npz files')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    parser.add_argument('y_min')
    parser.add_argument('y_max')
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

    try:
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
        ax.set_title('RQM: %s %s %s' % (platform, mode[0], mode[1]))
        ax.legend(('ESA', 'NERSC', 'Diff.'))
    except:
        pass

    plt.savefig(out_path, bbox_inches='tight', dpi=300)

def get_mean_std(pref, data):
    res_ll = []
    for key in data.keys():
        if pref in key:
            res_ll.append(data[key])
            print(data[key])
    return np.nanmean(np.concatenate(res_ll)), np.nanstd(np.concatenate(res_ll)), np.concatenate(res_ll)

def get_unique_regions(file_list):
    ''' Get unique combinations of mode, polarization and polarization mode '''
    ll = []
    for ifile in file_list:
        ll.append(os.path.basename(ifile).split('_')[-2])
    return list(set(ll))

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

for platform in platforms:
    for mode in modes:
        npz_list = glob.glob('%s/*%s*%s*%s*.npz' % (args.in_path, platform, mode[0], pol_mode[mode[1]]))
        unq_file_masks = sorted(get_unique_regions(npz_list))
        d_plot = {}

        for fmask in unq_file_masks:
            print('\n###%s###\n' % fmask)
            npz_list = glob.glob('%s/*%s*%s*%s*_%s*.npz' % (args.in_path, platform, mode[0], pol_mode[mode[1]], fmask))
            print(npz_list)
            print('###\n')
            total_esa_data = []
            total_nersc_data = []
            total_diff_data = []
            res_d = {}

            a = npz_list[0]
            a = np.load(a)
            d_npz = dict(zip((k for k in a), (a[k] for k in a)))

            for key in d_npz.keys():
                res_d['%s_ll' % key] = []

            for key in d_npz.keys():
                for npz in npz_list:
                    f_npz = np.load(npz)
                    d_npz = dict(zip((k for k in f_npz), (f_npz[k] for k in f_npz)))
                    var_name = '%s_ll' % key
                    res_d[var_name].append(d_npz[key])
                arr = np.concatenate(res_d[var_name])
                res_d[var_name] = arr

            d_plot[fmask] = {}
            d_plot[fmask]['Num_images'] = len(npz_list)
            d_plot[fmask]['Image_IDs'] = [os.path.basename(il).split('.')[0] for il in npz_list]

            m_esa, std_esa, data_esa = get_mean_std('ESA', res_d)
            d_plot[fmask]['Mean_ESA'] = m_esa
            d_plot[fmask]['STD_ESA'] = std_esa

            m_nersc, std_nersc, data_nersc = get_mean_std('NERSC', res_d)
            d_plot[fmask]['Mean_NERSC'] = m_nersc
            d_plot[fmask]['STD_NERSC'] = std_nersc

            diff = data_esa - data_nersc
            m_diff = np.nanmean(diff)
            std_diff = np.nanstd(diff)
            d_plot[fmask]['Mean_Diff'] = m_diff
            d_plot[fmask]['STD_Diff'] = std_diff

        plot_results(d_plot, '%s/%s_%s_%s_agg_plot.png' % (args.out_path, platform, mode[0], mode[1]))