#!/usr/bin/env python
""" This python script process individual S1 Level-1 GRD files
to get thermal noise removal quality assessment
(the range quality metric (RQM) and the azimuth quality metric(AQM))
for individual files in batch mode (each platform and sensing mode).
The script produces npz files.

run example:
python run_qm.py rqm /path/to/L1/GRD/files /path/to/output/dir
python run_qm.py aqm /path/to/L1/GRD/files /path/to/output/dir

"""
import argparse
import os
import glob
from multiprocessing import Pool
import numpy as np
from s1denoise import Sentinel1Image
from sentinel1calval import Sentinel1CalVal

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

    grd_mode = {
        'IW': 'GRDH',
        'EW': 'GRDM',
    }

    args = parse_run_experiment_args()
    os.makedirs(args.out_dir, exist_ok=True)

    platforms = ['S1A', 'S1B']
    regions = ['ARCTIC', 'ANTARCTIC', 'DESSERT', 'DOLLDRUMS', 'OCEAN']

    modes = [['IW', 'HV'],
             ['IW', 'VH'],
             ['EW', 'HV'],
             ['EW', 'VH']]

    qm_name = exp_names[args.quality_metric]

    out_dir = args.out_dir

    for platform in platforms:
        print('Processing %s' % platform)
        for mode in modes:
            pol = mode[1]
            for region in regions:
                qm_prefix = '%s_%s_%s' % (os.path.basename(args.inp_dir).split('_')[-1], region,
                                          args.quality_metric.upper())
                # find files for processing
                zip_files = sorted(glob.glob('%s/%s/%s/%s_%s_%s_%s_%s/*.zip' %
                                             (args.inp_dir, region, platform,
                                              mode[1], mode[0], grd_mode[mode[0]], platform, region)))
                # make directory for output npz files
                os.makedirs(args.out_dir, exist_ok=True)
                # launch proc in parallel
                with Pool(args.cores) as pool:
                    pool.map(run_process, zip_files)

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Batch quality assessment from individual S1 Level-1 GRD files')
    parser.add_argument('quality_metric', choices=['rqm', 'aqm'])
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
        s1 = Sentinel1CalVal(zipFile)
        func = getattr(s1, 'get_' + qm_name)
        res_qam = func(pol)
        print(res_qam)
        np.savez(out_fullname, **res_qam)

if __name__ == "__main__":
    main()