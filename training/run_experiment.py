#!/usr/bin/env python
""" This python script process individual S1 Level-1 GRD files
to get statistics for each sub-block

run example:
python run_experiment.py ns S1A VH /path/to/L1/GRD/files /path/to/output/dir
python run_experiment.py pb S1A VH /path/to/L1/GRD/files /path/to/output/dir

"""
import argparse
import os
import sys
import glob
import shutil
from multiprocessing import Pool


from s1denoise import Sentinel1Image

out_dir = None
pol = None
exp_name = None

exp_names = {
    'ns': 'noiseScaling',
    'pb': 'powerBalancing',
}


def main():
    """ Find zip files and launch (multi)processing """
    global out_dir, pol, exp_name
    args = parse_run_experiment_args()

    exp_name = exp_names[args.experiment]
    out_dir = args.out_dir
    pol = args.polarization

    # find files for processing
    zip_files = sorted(glob.glob(f'{args.inp_dir}/{args.platform}*.zip'))
    # make directory for output npz files
    os.makedirs(args.out_dir, exist_ok=True)
    # launch proc in parallel
    with Pool(args.cores) as pool:
        pool.map(run_process, zip_files)

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Aggregate statistics from individual NPZ files')
    parser.add_argument('experiment', choices=['ns', 'pb'])
    parser.add_argument('platform', choices=['S1A', 'S1B'])
    parser.add_argument('polarization', choices=['HV', 'VH'])
    parser.add_argument('inp_dir')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def run_process(zipFile):
    """ Process individual file with experiment_ """
    out_basename = os.path.basename(zipFile).split('.')[0] + f'_{exp_name}.npz'
    out_fullname = os.path.join(out_dir, out_basename)
    if os.path.exists(out_fullname):
        print(f'{out_fullname} already exists.')
    else:
        s1 = Sentinel1Image(zipFile)
        func = getattr(s1, 'experiment_' + exp_name)
        func(pol)
        print(f'Done! Moving file to {out_fullname}')
        shutil.move(out_basename, out_fullname)


if __name__ == "__main__":
    main()
