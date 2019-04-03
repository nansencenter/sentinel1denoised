#!/usr/bin/env python
import sys
import argparse

from s1denoise import Sentinel1Image
from s1denoise.utils import run_denoising

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
            description="Remove thermal noise from Sentinel-1 TOPSAR EW GRDM")
    parser.add_argument('ifile', type=str, help='input Sentinel-1 file in SAFE format')
    parser.add_argument('ofile', type=str, help='output GeoTIFD file')
    parser.add_argument('-db', '--decibel', action='store_true', help='Export in decibels')
    parser.add_argument('-nn', '--no-negatives', action='store_true',
                        help='Replace negative pixels with smallest positive pixel in the vicinity')
    parser.add_argument('-p', '--polarisation', default='HV',
                        choices=['HH', 'HV', 'HHHV',],
                        help='Process bands in these polarizations')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    pols = {'HH': ['HH'],
            'HV': ['HV'],
            'HHHV': ['HH', 'HV']}[args.polarisation]
    run_denoising(args.ifile, args.ofile,
                    pols=pols,
                    db=args.decibel,
                    filter_negative=args.no_negatives
                    )
