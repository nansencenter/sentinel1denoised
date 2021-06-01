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
from pathlib import Path

from s1denoise import Sentinel1Image

out_dir = None
pol = None
exp_name = None
force = False

exp_names = {
    'ns': 'noiseScaling',
    'pb': 'powerBalancing',
}


def main():
    """ Find zip files and launch (multi)processing """
    global out_dir, pol, exp_name, force
    args = parse_run_experiment_args()

    exp_name = exp_names[args.experiment]
    out_dir = args.out_dir
    pol = args.polarization
    force = args.force

    # find files for processing
    zip_files = sorted(args.inp_dir.glob(f'{args.platform}*.zip'))
    # make directory for output npz files
    args.out_dir.mkdir(exist_ok=True)
    
    # launch proc in parallel
    with Pool(args.cores) as pool:
        pool.map(run_process, zip_files)
    # run_process(zip_files[0])

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Aggregate statistics from individual NPZ files')
    parser.add_argument('experiment', choices=['ns', 'pb'])
    parser.add_argument('platform', choices=['S1A', 'S1B'])
    parser.add_argument('polarization', choices=['HV', 'HH'])
    parser.add_argument('inp_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('-c', '--cores', default=2, type=int,
                        help='Number of cores for parallel computation')
    parser.add_argument('--force', action='store_true', help="Force overwrite existing output")

    return parser.parse_args()

def run_process(zipFile):
    """ Process individual file with experiment_ """
    global out_dir, pol, exp_name, force

    default_output = zipFile.parent / (zipFile.stem + f'_{exp_name}.npz')
    desired_output = out_dir / default_output.name
    if desired_output.exists() and not force:
        print(f'{desired_output} already exists.')
    else:
        s1 = Sentinel1Image(zipFile.as_posix())
        func = getattr(s1, 'experiment_' + exp_name)
        func(pol)
        print(f'Done! Moving file to {desired_output}')
        with open(desired_output, 'wb') as handle:
            handle.write(default_output.read_bytes())
        default_output.unlink()


if __name__ == "__main__":
    main()
