#!/usr/bin/env python 
import sys
import argparse

import numpy as np
try:
    from nansat import Nansat
except ImportError:
    NANSAT_AVAILABLE = False
else:
    NANSAT_AVAILABLE = True

from s1denoise.tools import run_correction

def parse_args(args):
    ''' parse input arguments '''
    parser = argparse.ArgumentParser(
        description="Correct Sentinel-1 TOPSAR EW GRDM for thermal and texture noise and angular dependence")
    parser.add_argument('ifile', type=str, help='Input Sentinel-1 file in SAFE or zip format')
    parser.add_argument('ofile', type=str, help='Output file')
    parser.add_argument('-a', '--algorithm', 
                        help='Name of the algorithm to use', 
                        type=str, 
                        default='NERSC',
                        choices=['ESA', 'NERSC', 'NERSC_TG'])
    parser.add_argument('-g', '--geotiff', help='Export resulst as a GeoTIFF file', action='store_true')
    parser.add_argument('-m', '--mask', help='Also export land mask as a numpy file', action='store_true')
    return parser.parse_args(args)

def export_geotiff(ifile, ofile, d, mask):
    """ Use Nansat to export the arrays as a geotiff file with georeference as in the input dataset """
    remove_metadata = ['dataType', 'PixelFunctionType', 'SourceBand', 'SourceFilename', 'colormap', 'minmax', 'units', 'wkv']
    n_inp = Nansat(ifile)
    n_out = Nansat.from_domain(n_inp)
    for pol in d:
        parameters = n_inp.get_metadata(band_id='sigma0_%s' % pol)
        for i in remove_metadata:
            parameters.pop(i)
        n_out.add_band(array=d[pol], parameters=parameters)
        n_out.set_metadata(n_inp.get_metadata())    
    n_out.export(ofile, driver='GTiff')

def export_mask(ifile, ofile):
    """ Export landmask as a numpy file """
    n_inp = Nansat(ifile)
    wm = n_inp.watermask()
    np.savez_compressed(ofile + '_mask.npz', mask=wm[1])

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if (args.geotiff or args.mask) and not NANSAT_AVAILABLE:
        raise ImportError(' Nansat is not installed and I can\'t export geotif or landmask. '
                          ' Install Nansat or do not use "-g" and "-m". ')

    d = run_correction(args.ifile, algorithm=args.algorithm)
    if args.geotiff:
        export_geotiff(args.ifile, args.ofile, d, args.mask)
    else:
        np.savez_compressed(args.ofile, **d)
    if args.mask:
        export_mask(args.ifile, args.ofile)
