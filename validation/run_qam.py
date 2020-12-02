#!/usr/bin/env python
""" Range quality metric calculation

    run example: run run_qam.py polarization[HV/VH] input_path result_path

    output:
            npz file with RQM for ESA and NERSC algorithms

"""

import argparse
import glob
import json
import os
from s1denoise import Sentinel1Image
from multiprocessing import Pool

POL_MODES = {
    'HV': '1SDH',
    'VH': '1SDV',
}

def run_process(zipFile):
    """ Calculation of RQM for individual file """
    s1 = Sentinel1Image(zipFile)
    res_qam = s1.qualityAssesment(args.polarization)
    f_id = os.path.basename(zipFile).split('.')[0]
    out_d = {}
    out_d[f_id] = res_qam
    return out_d

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Range quality assessment from individual S1 ZIP or SAFE files')
    parser.add_argument('polarization', choices=['HV', 'VH'])
    parser.add_argument('s1_path')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--cores', default=3, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

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

# Dictonary for ESA and NERSC RQM results
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

# Launch proc in parallel
with Pool(args.cores) as pool:
    data = pool.map(run_process, ffiles)

d_data = {}
for i, idata in enumerate(data):
    for ikey in idata.keys():
        d_data[ikey] = idata[ikey]

for fname in ffiles:
    id_name = os.path.basename(fname).split('.')[0]
    for k in keys:
        d_res_esa[id_name][k].append(d_data[id_name]['QAM_ESA'][k])
        d_res_nersc[id_name][k].append(d_data[id_name]['QAM_NERSC'][k])

# Save resultant files with RQM for ESA and NERSC algorithms
with open(esa_out_fname, 'wt') as fesa, open(nersc_out_fname, 'wt') as fnersc:
    json.dump(d_res_esa, fesa)
    json.dump(d_res_nersc, fnersc)

