#!/usr/bin/env python

""" This python script collects values of antenna pattern gain (APG), sigma0, incidence angle, etc from pre-processed files.
Then it computes coefficient B for equation:
Y = X B

Where Y is sigma0_HV, and X is a mtrix with the following columns:
[1, incidence_angel, apg_ew1, 1, apg_ew2, 1, apg_ew3, 1, apg_ew4, 1, apg_ew5, 1]
The first two columns have values for all rows, other columns have values only for correposning sub-swaths, or zeros for other subswaths.

Coefficient B is computed for each combination of platform, polarisation, mode, IPF
The results are saved in denoising_parameters.json

run example:
python process_apg_data.py /path/to/APG-files

"""
import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from s1denoise.utils import skip_swath_borders, build_AY_matrix, solve
from update_parameter_file import safe_load

array_names = [
    'line',
    'pixel',
    'noise',
    'swath_ids',
    'eap',
    'rsl',
    'eleang',
    'incang',
    'cal_s0hv',
    'scall_hv',
    'dn_vectors',
]

item_names = [
    'pg_amplitude',
    'noise_po_co_fa',
    'k_proc',
    'noise_ca_fa',
    'ipf',
]

polarization = 'HV'
scale_APG = 1e11
scale_HV = 1000
s0hv_max = [None, 3.0, 1.3, 1.0, 0.9, 0.8]
s0hv_apg_corr_min = [None, 0.96, 0.89, 0.89, 0.95, 0.80]
IPF_mapping = {
    2.36 : 2.36,
    2.40 : 2.43, # extra
    2.43 : 2.43,
    2.45 : 2.45,
    2.50 : 2.51, # extra
    2.51 : 2.51,
    2.52 : 2.52,
    2.53 : 2.53,
    2.60 : 2.60,
    2.70 : 2.71, # extra
    2.71 : 2.71,
    2.72 : 2.72,
    2.82 : 2.82,
    2.84 : 2.84,
    2.90 : 2.91, # extra
    2.91 : 2.91,
    3.10 : 3.10,
    3.20 : 3.31, # extra
    3.30 : 3.31, # extra
    3.31 : 3.31,
    3.40 : 3.52, # extra
    3.51 : 3.52, # extra
    3.52 : 3.52,
    3.61 : 3.61,
}

def get_uid_mapping(uids):
    uid_mapping = {}
    for platform in ['S1A', 'S1B']:
        ipf2uid = {float(uid.split('_')[-1]): uid for uid in uids if uid.startswith(platform)}
        for dst_ipf in IPF_mapping:
            src_ipf = IPF_mapping[dst_ipf]
            if src_ipf in ipf2uid:
                src_uid = ipf2uid[src_ipf]
                dst_uid = '_'.join(src_uid.split('_')[:-1] + [f'{dst_ipf:04.2f}'])
                uid_mapping[dst_uid] = src_uid
    return uid_mapping

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Process SAFE or ZIP files and collect APG related values')
    parser.add_argument('inp_dir', type=Path)
    parser.add_argument('out_file', type=Path)
    return parser.parse_args()


args = parse_run_experiment_args()
ifiles = sorted(glob.glob(f'{args.inp_dir}/S1*_apg.npz'))

l = defaultdict(list)
for ifile in ifiles:
    print('Read', ifile)
    # load data
    ds = np.load(ifile, allow_pickle=True)
    d = {n: ds[n] for n in array_names}
    d.update({n: ds[n].item() for n in item_names})

    # compute values
    sigma0hv = d['dn_vectors'] ** 2 / d['cal_s0hv'] ** 2
    g_tots = []
    for i, (jjj, eee, rrr, sss, ccc) in enumerate(zip(d['swath_ids'], d['eap'], d['rsl'], d['scall_hv'], d['cal_s0hv'])):
        g_tot = sss / eee ** 2 / rrr ** 2 / ccc ** 2
        for j in range(1,6):
            gpi = jjj == j
            key = f'EW{j}'
            g_tot[gpi] *= d['k_proc'][key]
            g_tot[gpi] *= d['noise_ca_fa'][key]
            g_tot[gpi] *= d['pg_amplitude'][key][i]
            g_tot[gpi] *= d['noise_po_co_fa'][key][i]
        g_tots.append(g_tot)

    l['ipf'].append(d['ipf'])
    l['apg'].append(np.array(g_tots[1:-1]))
    l['sigma0hv'].append(sigma0hv[1:-1])
    l['swath_ids'].append(d['swath_ids'][1:-1])
    l['incang'].append(d['incang'][1:-1])

ll = defaultdict(list)
for ipf, apg, sigma0hv, swath_ids, incang, ifile in zip(l['ipf'], l['apg'], l['sigma0hv'], l['swath_ids'], l['incang'], ifiles):
    print('Build', ifile)
    name_parts = os.path.basename(ifile).split('_')
    platform, mode, resolution, pol = name_parts[0], name_parts[1], name_parts[2], name_parts[3]
    uid = f'{platform}_{mode}_{resolution}_{pol}_APG_{ipf:04.2f}'
    swath_ids_skip = skip_swath_borders(swath_ids, skip=2)
    sigma0hv_s = [i * scale_HV for i in sigma0hv]
    apg_s = [i * scale_APG for i in apg]
    A, Y = build_AY_matrix(swath_ids_skip, sigma0hv_s, apg_s, incang, s0hv_max, s0hv_apg_corr_min)
    if A is not None:
        ll['a'].append(A)
        ll['y'].append(Y)
        ll['uid'].append(uid)

uids, uid_inverse = np.unique(ll['uid'], return_inverse=True)
B = {}
rmsd = {}

for i, uid in enumerate(uids):
    print('Solve', uid)
    uid_indices = np.where(uid_inverse == i)[0]
    A = [ll['a'][uid_idx] for uid_idx in uid_indices]
    Y = [ll['y'][uid_idx] for uid_idx in uid_indices]
    A = np.vstack(A)
    Y = np.vstack(Y)
    B[uid], rmsd[uid] = solve(A, Y)
    Yrec = np.dot(A, B[uid])
    plt.plot(Y.flat, Yrec, 'k.', alpha=0.1)
    plt.plot(Y.flat, Y.flat, 'r-')
    plt.title(f'{uid} {rmsd[uid]:1.2f}')
    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.gca().set_aspect('equal')
    plt.savefig(f'{uid}_quality_pg.png')
    plt.close()

uid_mapping  = get_uid_mapping(uids)

p = safe_load(args.out_file)
for uid in uid_mapping:
    print(f'Save {uid} ({uid_mapping[uid]})')
    p[uid] = dict(
        Y_SCALE = scale_HV,
        A_SCALE = scale_APG,
        B = B[uid_mapping[uid]].tolist(),
        RMSD = rmsd[uid_mapping[uid]],
    )
    
with open(args.out_file, "w") as f:
    json.dump(p, f)
