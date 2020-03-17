#!/usr/bin/env python
import numpy as np
from scipy.ndimage import minimum_filter
from nansat import Nansat

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
        array = 10 * np.log10(array) + scale[pol] * (inc - angular_offset)
        n.add_band(array=array.astype(output_dtype), parameters=parameters)
    n.set_metadata(s1.get_metadata())
    return n
