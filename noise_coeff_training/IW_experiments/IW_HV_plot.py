""" Test plotting for IW thermal denoise examples
    Last modified: 23-06-2020 by DD
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from s1denoise.S1_TOPS_GRD_NoiseCorrection import Sentinel1Image
import cv2
import numpy as np
import os
import glob
import sys

def norm8(img):
    """Convert image to data type uint8"""
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

def format_func(value, tick_number):
    # find number of multiples of pi/2
    try:
        N = '%2.0f' % ia[int(value)]
    except:
        N = '%2.0f' % ia[-1]
    return N

def plot_png_denoised(data, ia, pref):
    """ Plot denoised png """
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel('Incidence angle')
    plt.ylabel('Azimuth line')
    plt.title(pref, fontsize='medium')
    plt.imshow(data, cmap='gray', interpolation='bilinear')
    # list comprehension to get all tick labels...
    tickla = ['%2.0f' % tick for tick in ia]
    ax.xaxis.set_ticklabels(tickla)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.savefig('%s/%s_%s.png' % (outfile_path, pref, os.path.basename(fname)),
                bbox_inches='tight', vmin=0, cmax=255, dpi=150)

#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/IW_GRD_HD_20190627-20200608/S1A_IW_GRDH_1SDH_20190715T174645_20190715T174714_028131_032D65_7473.zip'

# Test independent data
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/IW_GRD_HD_20190627-20200608/independent/S1A_IW_GRDH_1SDH_20200608T095534_20200608T095559_032924_03D057_0E54.zip'

infile_path = sys.argv[1]
outfile_path = sys.argv[2]

ffiles = glob.glob('%s/*.zip' % infile_path)

for fname in ffiles:

    n = Sentinel1Image(fname)
    s0 = n.rawSigma0Map(polarization='HV')

    ia = np.nanmean(n['incidence_angle'], axis=0)

    esa_nesz = n.rawNoiseEquivalentSigma0Map(polarization='HV')
    nersc_nesz = n.modifiedNoiseEquivalentSigma0Map(polarization='HV')

    denoised_nersc = s0 - nersc_nesz
    denoised_nersc = 10 * np.log10(denoised_nersc)

    denoised_esa = s0 - esa_nesz
    denoised_esa = 10 * np.log10(denoised_esa)

    data_nersc = norm8(denoised_nersc)
    data_esa = norm8(denoised_esa)

    # Plot denoised images
    plot_png_denoised(data_esa, ia, 'ESA')
    plot_png_denoised(data_nersc, ia, 'NERSC')