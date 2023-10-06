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

array_names = ['line',
    'pixel',
    'noise',
    'swath_ids',
    'eap',
    'rsl',
    'eleang',
    'incang',
    'cal_s0hv',
    'scall_hv',
    'sigma0hv',
]

item_names = [
    'pgpp',
    'pgpa',
    'kproc',
    'ncf',
    'acc',
    'ipf'
]

polarization = 'HV'
scale_APG = 1e21
scale_HV = 1000
s0hv_max = [None, 3.0, 1.3, 1.0, 0.9, 0.8]


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
    try:
        ds = np.load(ifile, allow_pickle=True)
        d = {n: ds[n] for n in array_names}
    except:
        # TEMPORARY!
        # skip if file is written at the moment
        continue

    d.update({n: ds[n].item() for n in item_names})
    sigma0hv = d['sigma0hv'] ** 2 / d['cal_s0hv'] ** 2
    apg = (1 / d['eap'] / d['rsl']) ** 2 / d['cal_s0hv'] ** 2 * d['scall_hv']
    l['ipf'].append(d['ipf'])
    l['apg'].append(apg[1:-1])
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
    A, Y = build_AY_matrix(swath_ids_skip, sigma0hv_s, apg_s, incang, s0hv_max)
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
    plt.savefig(f'{uid}_{hv_name}_quality.png')
    plt.close()

p = safe_load(args.out_file)
for uid in B:
    print('Save', uid)
    p[uid] = dict(
        Y_SCALE = scale_HV,
        A_SCALE = scale_APG,
        B = B[uid].tolist(),
        RMSD = rmsd[uid],
    )
    
with open(args.out_file, "w") as f:
    json.dump(p, f)
