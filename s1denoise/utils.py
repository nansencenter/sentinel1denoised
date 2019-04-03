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

def run_denoising(ifile, ofile,
                    pols=['HV'],
                    db=False,
                    filter_negative=False,
                    **kwargs):
    """ Run denoising of input file

    Parameters
    ----------
    ifile : str
        input file
    ofile : str
        output file
    pols : str
        polarisoation options, ['HH'], ['HV'] or ['HH','HV']
    db : bool
        convert to decibel?
    data_format : str
        format of data in output file
    filter_negative : bool
        replace negative values with smallest nearest positive?

    Modifies
    --------
    Writes to the output file in GeoTIFF format

    """
    s1 = Sentinel1Image(ifile)
    n = Nansat.from_domain(s1)
    for pol in pols:
        s1.add_denoised_band(pol)
        array = s1['sigma0_%s_denoised' % pol]
        parameters = s1.get_metadata(band_id='sigma0_%s' % pol)
        if filter_negative:
            array = remove_negative(array)
        if db:
            array = 10 * np.log10(array)
            parameters['units'] = 'dB'
        n.add_band(array=array,
                   parameters=parameters)
    n.set_metadata(s1.get_metadata())
    n.export(ofile, driver='GTiff')
