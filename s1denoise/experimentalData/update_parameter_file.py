#!/usr/bin/env python
""" Copy noise scaling or power balancing coefficients to the main JSON file

python update_parameter_file.py powerBalancing/S1B_EW_GRDM_1SDH_power_balancning.json ../denoising_parameters.json
python update_parameter_file.py noiseScaling/S1B_EW_GRDM_1SDH_noise_scaling.json ~/s1denoise/parameters.json
"""

import argparse
import json
import os

import numpy as np

def parse_args():
    """ Parse input args for analyze_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Copy coefficients to central JSON file')
    parser.add_argument('inp_file')
    parser.add_argument('out_file')
    return parser.parse_args()

def safe_load(input_file):
    """ Load JSON file or make empty dict """
    with open(input_file, 'rt') as f:
        try:
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
        if ipf_key not in out_files:
            out_files[ipf_key] = {}
        for swath in inp_par[ipf_key]['mean']:
            print(ipf_key, swath, inp_par[ipf_key]['mean'][swath], out_par[ipf_key][swath])
            out_par[ipf_key][swath] = inp_par[ipf_key]['mean'][swath]
            out_files[ipf_key] = inp_par[ipf_key]['files']

    with open(args.out_file, 'w') as f:
        json.dump(out_par, f)

    with open(out_file2, 'w') as f:
        json.dump(out_files, f)

if __name__ == "__main__":
    main()

# how to convert the old JSON files to the new format
"""
for platform in ['S1A', 'S1B']:
    ifile = f'../denoising_parameters_{platform}.json'

    with open(ifile, 'rt') as f:
        old_par = json.load(f)

    param_types = {
        'powerBalancingParameters': 'PB',
        'noiseScalingParameters': 'NS',
        'noiseVarianceParameters': 'NV',
        'extraScalingParameters': 'ES',
    }

    swath2res = {
        'IW': 'GRDH',
        'EW': 'GRDM',
    }

    new_par = {}
    for pol in old_par:
        for tp in old_par[pol]:
            param_type = param_types[tp]
            print(param_type)
            if param_type in ['NS', 'PB']:
                for swath in old_par[pol][tp]:
                    for ipf in old_par[pol][tp][swath]:
                        mode = swath[:2]
                        res = swath2res[mode]
                        new_key = f'{platform}_{mode}_{res}_{pol}_{param_type}_{ipf}'
                        if new_key not in new_par:
                            new_par[new_key] = {}
                        new_par[new_key][swath] = old_par[pol][tp][swath][ipf]
            elif param_type in ['NV', 'ES']:
                ipf = '2.9'
                mode = 'EW'
                res = swath2res[mode]
                for swath in old_par[pol][tp]:
                    new_key = f'{platform}_{mode}_{res}_{pol}_{param_type}_{ipf}'
                    if new_key not in new_par:
                        new_par[new_key] = {}
                    new_par[new_key][swath] = old_par[pol][tp][swath]

with open('../../denoising_parameters.json', 'w') as f:
    json.dump(new_par, f)


"""
