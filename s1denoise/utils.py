#!/usr/bin/env python
import argparse
import datetime
import glob
import json
import os

from nansat import Nansat
import numpy as np
from scipy.ndimage import minimum_filter

from s1denoise import Sentinel1Image

def remove_negative(array, window=10, **kwargs):
    """ Replace negative values with lowest positive from small vicinity

    Parameters
    ----------
    array : ndarray
        array with gaps with negative pixels
    window : int
        window size to search for the closest positive value

    Returns
    -------
    array : ndarray
        gaps with negative pixels are filed with smallest positive

    """
    mask = array <= 0
    arr2 = np.array(array)
    arr2[mask] = +np.inf
    arr2 = minimum_filter(arr2, window)
    array[mask] = arr2[mask]
    return array

def run_denoising(ifile, pols=['HV'], db=False, filter_negative=False, **kwargs):
    """ Run denoising of input file

    Parameters
    ----------
    ifile : str
        input file
    pols : str
        polarisoation options, ['HH'], ['HV'] or ['HH','HV']
    db : bool
        convert to decibel?
    data_format : str
        format of data in output file
    filter_negative : bool
        replace negative values with smallest nearest positive?

    Returns
    --------
    s1 : Nansat
        object with denoised bands and metadata

    """
    s1 = Sentinel1Image(ifile)
    n = Nansat.from_domain(s1)
    for pol in pols:
        print('Denoise %s band' % pol)
        s1.add_denoised_band(pol)
        array = s1['sigma0_%s_denoised' % pol]
        parameters = s1.get_metadata(band_id='sigma0_%s' % pol)
        if filter_negative:
            print('Remove negaive pixels')
            array = remove_negative(array)
        if db:
            print('Convert to dB')
            array = 10 * np.log10(array)
            parameters['units'] = 'dB'
        n.add_band(array=array, parameters=parameters)
    n.set_metadata(s1.get_metadata())

    return n

def run_correction(ifile,
    angular_scale_hh=-0.2,
    angular_scale_hv=-0.025,
    angular_offset=34.5,
    output_dtype=np.float32):
    """ Run thermal, textural and angular correction of input Sentinel-1 file

    Parameters
    ----------
    ifile : str
        input file
    angular_scale_hh : float
        Scale for angular correction of sigma0 in HH
    angular_scale_hv : float
        Scale for angular correction of sigma0 in HV
    angular_offset : float
        Central angle for sigma0 normalization
    output_dtype : dtype
        Type of output array

    Returns
    --------
    s1 : Nansat
        object with corrected bands and metadata

    """
    scale = {
        'HH': angular_scale_hh,
        'HV': angular_scale_hv,
    }

    pols = ['HH', 'HV']
    s1 = Sentinel1Image(ifile)
    n = Nansat.from_domain(s1)
    inc = s1['incidence_angle']
    for pol in pols:
        print('Correct %s band' % pol)
        parameters = s1.get_metadata(band_id='sigma0_%s' % pol)
        for i in ['dataType', 'PixelFunctionType', 'SourceBand', 'SourceFilename']:
            parameters.pop(i)
        array = s1.texturalNoiseRemoval2(pol)
        array = 10 * np.log10(array) - scale[pol] * (inc - angular_offset)
        n.add_band(array=array.astype(output_dtype), parameters=parameters)
    n.set_metadata(s1.get_metadata())
    return n


class AnalyzeExperiment():
    def parse_analyze_experiment_args(self):
        """ Parse input args for analyze_experiment_* scripts """
        parser = argparse.ArgumentParser(description='Aggregate statistics from individual NPZ files')
        parser.add_argument('platform', choices=['S1A', 'S1B'])
        parser.add_argument('mode', choices=['EW', 'IW'])
        parser.add_argument('res', choices=['GRDM', 'GRDH'])
        parser.add_argument('pol', choices=['1SDH', '1SDV'])
        parser.add_argument('inp_dir')
        parser.add_argument('out_dir')
        return parser.parse_args()

    def __init__(self):
        """ Initialize """
        self.args = self.parse_analyze_experiment_args()

        # dicts with sub-swaths number and polarization
        self.swaths_number = {'IW': 3, 'EW': 5}[self.args.mode]
        self.swath_names = ['%s%s' % (self.args.mode,iSW) for iSW in range(1, self.swaths_number+1)]
        self.polarisation = {'1SDH':'HV', '1SDV':'VH'}[self.args.pol]

        self.out_filename = os.path.join(
            self.args.out_dir,
            f'{self.args.platform}_{self.args.mode}_{self.args.res}_{self.args.pol}_{self.file_suffix}.json'
        )

        npzFilesAll = sorted(glob.glob(os.path.join(self.args.inp_dir,
            f'{self.args.platform}_{self.args.mode}_{self.args.res}_{self.args.pol}_*_{self.file_suffix}.npz')))

        # Check quality disclaimer #30 and #31 in https://qc.sentinel1.eo.esa.int/disclaimer/
        self.npzFiles = []
        for li, npzFile in enumerate(npzFilesAll):
            print(npzFile)
            startDateTime = datetime.datetime.strptime(os.path.basename(npzFile).split('/')[-1][17:32], "%Y%m%dT%H%M%S")
            endDateTime = datetime.datetime.strptime(os.path.basename(npzFile).split('/')[-1][33:48], "%Y%m%dT%H%M%S")
            if (     self.args.platform=='S1A'
                 and startDateTime >= datetime.datetime(2018,3,13,1,0,42)
                 and endDateTime <= datetime.datetime(2018,3,15,14,1,26) ):
                continue
            elif (     self.args.platform=='S1B'
                   and startDateTime >= datetime.datetime(2018,3,13,2,43,5)
                   and endDateTime <= datetime.datetime(2018,3,15,15,19,30) ):
                continue
            else:
                self.npzFiles.append(npzFile)

    def save(self, results):
        """ Save results in JSON file """
        with open(self.out_filename, 'wt') as f:
            json.dump(results, f)
