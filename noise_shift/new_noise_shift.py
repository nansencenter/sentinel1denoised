from s1denoise import Sentinel1Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from s1denoise.S1_TOPS_GRD_NoiseCorrection import Sentinel1Image
from multiprocessing import Pool
import sys
import glob
import os
import cv2
from numpy import inf

from scipy.ndimage import uniform_filter1d


def proc(s1_filename):
    n = Sentinel1Image('%s'
                       % (s1_filename))
    results = {}
    results['src'] = os.path.basename(s1_filename)
    results['inc'] = np.nanmean(n.incidenceAngleMap(polarization='HV'), axis=0)
    sz = n.rawSigma0Map(polarization='HV')
    sz[sz==0] = np.nan
    results['sz'] = np.nanmean(sz, axis=0)
    results['nesz_esa'] = np.nanmean(n.rawNoiseEquivalentSigma0Map(polarization='HV'), axis=0)
    results['nesz_nersc'] = np.nanmean(n.modifiedNoiseEquivalentSigma0Map(polarization='HV'), axis=0)

    return results

def denoise_parameters(s1_filename):
    n = Sentinel1Image('%s'
                       % (s1_filename))
    #results = {}
    s0 = n.rawSigma0Map(polarization='HV')
    nesz_esa = n.rawNoiseEquivalentSigma0Map(polarization='HV')
    nesz_nersc = n.modifiedNoiseEquivalentSigma0Map(polarization='HV')

    return s0, nesz_esa, nesz_nersc

def plot_burst(s0_hv):
    plt.clf()
    data_plot = s0_hv[swath_bounds[subswath_name]['firstAzimuthLine'][i]:
                      swath_bounds[subswath_name]['lastAzimuthLine'][i],
                swath_bounds[subswath_name]['firstRangeSample'][i]:
                swath_bounds[subswath_name]['lastRangeSample'][i]]
    s0_hv_ss = 10 * np.log10(data_plot)
    plt.imshow(s0_hv_ss, cmap='gray', vmin=-50, vmax=-5)
    plt.savefig('s0/s0_hv_%s_%s.png' % (subswath_name, i), bbox_inches='tight')

def norm8(img):
    """Convert image to data type uint8"""
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

try:
    os.makedirs('noise')
except:
    pass

try:
    os.makedirs('s0')
except:
    pass

try:
    os.makedirs('ncc')
except:
    pass

#fname = "/mnt/sverdrup-2/sat_auxdata/denoise/north_atl/s1/ew/zip/201806260830-201906261000/S1A_EW_GRDM_1SDH_20180629T065044_20180629T065142_022568_0271D6_5131.zip"

#fname = "/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20190907T062508_20190907T062608_017928_021BE5_4887.zip"
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/201906261000-202002041121/S1A_EW_GRDM_1SDH_20190929T073105_20190929T073205_029233_035224_2930.zip'
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20191130T062509_20191130T062609_019153_024264_9EB7.zip'

# 5E 70N, low signal
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20190802T062506_20190802T062606_017403_020BA1_DB2C.zip'
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20191118T062509_20191118T062609_018978_023CD7_4FF4.zip'
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20190721T062506_20190721T062606_017228_020668_8DAE.zip'
#fname = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/desc_5E_70N_201907010000-2020012800/S1B_EW_GRDM_1SDH_20191013T062509_20191013T062609_018453_022C34_0DED.zip'

# Noise shift case old IPF
fname = '/mnt/sverdrup-2/sat_auxdata/denoise/noise_shift_case/S1A_EW_GRDM_1SDH_20180629T065044_20180629T065142_022568_0271D6_5131.zip'

plot_denoised_figures = False

n  = Sentinel1Image(fname)

ia = n['incidence_angle']
s0_hv = n['sigma0_HV']
noise_hv = n['noise_HV']

plt.clf()
s0_hv_db = 10 * np.log10(s0_hv)
plt.imshow(s0_hv_db, cmap='gray', vmin=-50, vmax=-5)
plt.savefig('s0_hv.png', bbox_inches='tight')

swath_bounds = n.import_swathBounds('HV')

for li in range(1, {'IW':3, 'EW':5}[n.obsMode]+1):
    print('\n\n%s%s' % (n.obsMode, li))
    subswath_name = '%s%s' % (n.obsMode, li)
    #globals()[var_name] = var_name

    plt.clf()

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    fig.set_figheight(21)
    fig.set_figwidth(15)

    fig.suptitle('%s%s. Normilized cross correlation (NCC) between Sigma0 and NESZ'
                 % (n.obsMode, li), fontsize=24)

    for i in range(len(swath_bounds[subswath_name]['firstAzimuthLine'])):

        s0 = s0_hv[swath_bounds[subswath_name]['firstAzimuthLine'][i]:
                     swath_bounds[subswath_name]['lastAzimuthLine'][i],
                     swath_bounds[subswath_name]['firstRangeSample'][i]:
                     swath_bounds[subswath_name]['lastRangeSample'][i]]

        nesz = noise_hv[swath_bounds[subswath_name]['firstAzimuthLine'][i]:
                         swath_bounds[subswath_name]['lastAzimuthLine'][i],
                swath_bounds[subswath_name]['firstRangeSample'][i]:
                swath_bounds[subswath_name]['lastRangeSample'][i]]

        # Convert signal and noise to db
        s0 = 10*np.log10(s0)
        nesz = 10*np.log10(nesz)
        s0[s0==-inf] = np.nan
        nesz[nesz==-inf] = np.nan

        #plot_burst(s0)

        '''
        plt.clf()
        s0_hv_db = 10 * np.log10(s0)
        plt.imshow(s0_hv_db, cmap='gray', vmin=-50, vmax=-5)
        plt.savefig('s0/s0_%s_%s.png' % (subswath_name, i), bbox_inches='tight')
        '''


        # normilized cross-correlation of noise and sigma0

        # signal row
        middle_row = round(s0.shape[0]/2)

        # average rows
        num_rows = 25

        # cut rows
        cut_col = 200

        # signal rows
        #a = np.nanmean(s0[middle_row-num_rows:middle_row+num_rows, cut_col:-cut_col:], axis=0)
        # without cut
        s0_db_roi = s0[middle_row - num_rows:middle_row + num_rows,:]
        a = np.nanmean(s0_db_roi, axis=0)
        #a = uniform_filter1d(a, 10)

        # noise rows
        #b = np.nanmean(nesz[middle_row-num_rows:middle_row+num_rows, cut_col:-cut_col:], axis=0)
        # without cut
        nesz_db_roi = nesz[middle_row - num_rows:middle_row + num_rows, :]
        b = np.nanmean(nesz_db_roi, axis=0)
        #b = uniform_filter1d(b, 10)

        a = (a - np.nanmean(a)) / (np.nanstd(a) * len(a))
        b = (b - np.nanmean(b)) / (np.nanstd(b))
        c = np.correlate(a, b, 'full')

        # peak position
        ij = np.unravel_index(np.argmax(c), c.shape)

        print('\n%s' % a[np.isnan(a)==False])
        #print('%s\n' % s0_db_roi)
        print('\n%s' % b[np.isnan(b)==False])
        #print('%s\n' % nesz_db_roi)

        print('max cc: %.2f  peak pos.: %s  length: %s' %
              (np.nanmax(c), ij, len(c)))


        #plt.title('Normilized cross correlation (NCC) between Sigma0 and NESZ')


        #if i == 7:
        #    var_name = 'ax%s' % (i-1)
        #else:
        var_name = 'ax%s' % (i+1)


        #var_name = 'ax%s' % i

        # = var_name
        try:
            globals()[var_name].set_title('Burst %s' % (i+1), fontsize=20)
            globals()[var_name].set_ylim([0,1])
            globals()[var_name].plot(range(len(c)), c, 'b-', label='NCC')

            # Obtained peak position
            globals()[var_name].plot([ij, ij], [np.nanmin(c), np.nanmax(c)],
                                     'r--', label = 'NCC peak pos.')

            # True peak position
            globals()[var_name].plot([len(c)/2., len(c)/2.], [np.nanmin(c), np.nanmax(c)], 'k--', label='The true center')

            globals()[var_name].legend(loc=2, prop=dict(size=14))

            # calculate shift
            s0_nesz_shift = abs(len(c)/2. -  ij[0])
            if s0_nesz_shift < 1.:
                s0_nesz_shift = 0.

            s0_nesz_shift = round(s0_nesz_shift)

            anchored_text = AnchoredText('shift=%.0f px' % s0_nesz_shift, loc=1,
                                         prop=dict(fontweight="bold",fontsize=20))
            globals()[var_name].add_artist(anchored_text)
        except:
            pass

    plt.subplots_adjust(hspace=0.3)
    plt.savefig('ncc/EW%s_ncc_%s.png' % (li, os.path.basename(fname)), bbox_inches='tight')