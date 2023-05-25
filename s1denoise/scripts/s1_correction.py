#!/usr/bin/env python 
import sys
import argparse

import numpy as np

from s1denoise.tools import run_correction

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
        description="Correct Sentinel-1 TOPSAR EW GRDM for thermal and texture noise and angular dependence")
    parser.add_argument('ifile', type=str, help='input Sentinel-1 file in SAFE format')
    parser.add_argument('ofile', type=str, help='output GeoTIFF file')
    parser.add_argument('-m', '--mask', help='Export land mask as a numpy file', action='store_true')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    s1 = run_correction(args.ifile)
    s1.export(args.ofile, driver='GTiff')
    if args.mask:
        wm = s1.watermask()
        np.savez_compressed(args.ofile + '_mask.npz', mask=wm[1])
