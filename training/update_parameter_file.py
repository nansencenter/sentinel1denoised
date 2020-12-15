#!/usr/bin/env python
""" Copy noise scaling or power balancing coefficients to the main JSON file

python update_parameter_file.py powerBalancing/S1B_EW_GRDM_1SDH_power_balancning.json ../denoising_parameters.json
python update_parameter_file.py noiseScaling/S1B_EW_GRDM_1SDH_noise_scaling.json ~/s1denoise/parameters.json
"""

import argparse
import json
import os

import numpy as np

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 100.0]"%(x,))
    return x

def parse_args():
    """ Parse input args for analyze_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Copy coefficients to central JSON file')
    parser.add_argument('inp_file')
    parser.add_argument('out_file')
    parser.add_argument('-d', '--dst_ipf', type=restricted_float,
                        help='Additional destination IPF ver. to copy the results')
    return parser.parse_args()

def safe_load(input_file):
    """ Load JSON file or make empty dict """
    try:
        with open(input_file, 'rt') as f:
            result = json.load(f)
    except:
        print(f'Invalid data in {input_file}')
        result = {}
    return result

def main():
    args = parse_args()
    out_file2 = args.out_file.replace('.json', '_training_files.json')
    ifiles = [args.inp_file, args.out_file, out_file2]
    inp_par, out_par, out_files = [safe_load(ifile) for ifile in ifiles]

    for ipf_key in inp_par:
        if ipf_key not in out_par:
            out_par[ipf_key] = {}
            if args.dst_ipf is not None:
                dst_ipf_key = ipf_key.replace(ipf_key.split('_')[-1], str(args.dst_ipf))
                out_par[dst_ipf_key] = {}
        if ipf_key not in out_files:
            out_files[ipf_key] = {}
            if args.dst_ipf is not None:
                out_files[dst_ipf_key] = {}
        for swath in inp_par[ipf_key]['mean']:
            #print(ipf_key, swath, inp_par[ipf_key]['mean'][swath], out_par[ipf_key][swath])
            out_par[ipf_key][swath] = inp_par[ipf_key]['mean'][swath]
            out_files[ipf_key] = inp_par[ipf_key]['files']
            if args.dst_ipf is not None:
                #print(args.dst_ipf, swath, inp_par[ipf_key]['mean'][swath], out_par[ipf_key][swath])
                out_par[dst_ipf_key][swath] = inp_par[ipf_key]['mean'][swath]
                out_files[dst_ipf_key] = inp_par[ipf_key]['files']

    with open(args.out_file, 'w') as f:
        json.dump(out_par, f)

    with open(out_file2, 'w') as f:
        json.dump(out_files, f)

if __name__ == "__main__":
    main()
