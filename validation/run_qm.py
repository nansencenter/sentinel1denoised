#!/usr/bin/env python
""" This python script process individual S1 Level-1 GRD files
to get thermal noise removal quality assessment
(the range quality metric (RQM) and the azimuth quality metric(AQM))
for individual files

run example:
python run_qm.py rqm S1A VH /path/to/L1/GRD/files /path/to/output/dir
python run_qm.py aqm S1A VH /path/to/L1/GRD/files /path/to/output/dir

"""
import argparse
import os
import glob
from multiprocessing import Pool
import numpy as np
from s1denoise import Sentinel1Image

out_dir = None
pol = None
exp_name = None

exp_names = {
    'rqm': 'range_quality_metric',
    'aqm': 'azimuth_quality_metric',
}

def main():
    """ Find zip files and launch (multi)processing """
    global out_dir, pol, qm_name, qm_prefix
    args = parse_run_experiment_args()

    qm_name = exp_names[args.quality_metric]
    qm_prefix = '%s_%s' % (os.path.basename(args.inp_dir).split('_')[-1], args.quality_metric.upper())
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
    parser = argparse.ArgumentParser(description='Quality assessment from individual S1 files')
    parser.add_argument('quality_metric', choices=['rqm', 'aqm'])
    parser.add_argument('platform', choices=['S1A', 'S1B'])
    parser.add_argument('polarization', choices=['HV', 'VH'])
    parser.add_argument('inp_dir')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--cores', default=3, type=int,
                        help='Number of cores for parallel computation')
    return parser.parse_args()

def run_process(zipFile):
    """ Process individual file with get_quality_metric """
    out_basename = os.path.basename(zipFile).split('.')[0] + f'_{qm_prefix}.npz'
    out_fullname = os.path.join(out_dir, out_basename)

    if os.path.exists(out_fullname):
        print(f'{out_fullname} already exists.')
    else:
        s1 = Sentinel1Image(zipFile)
        func = getattr(s1, 'get_' + qm_name)
        res_qam = func(pol)
        print(res_qam)
        np.savez(out_fullname, **res_qam)

if __name__ == "__main__":
    main()
