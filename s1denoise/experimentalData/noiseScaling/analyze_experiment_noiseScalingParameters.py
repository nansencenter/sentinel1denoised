import os
import sys
import glob
import datetime
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import exit
from collections import defaultdict

#########################################################################################################
# run example:
# run analyze_experiment_noiseScalingParameters.py IW GRDH 1SDV /path/to/npz /path/to/output/file
#
# '/path/to/output/file' is a resultant file with updated/replaced coefficients for noise scaling
# which you can replace the old one
#
# Output file name will be as following: 'ns_MODE_GRD-MODE_denoising_parameters_PLATFORM.npz' and
# contains coefficients for previous IPF versions and modes plus updated data
#
# It is important to define a correct path to your existing npz file with coefficients in
# variable called 'path_to_coefficients_npz'
#########################################################################################################

# Instrument
platform = sys.argv[1]

# 1st define path to your local existing file with coefficients that is suppose to be the basis for the updated file in output dir
#path_to_coefficients_npz = os.path.join(os.path.dirname(os.path.realpath(__file__))
path_to_coefficients_npz = '/Home/denemc/miniconda3/envs/py3s1denoise/lib/python3.7/site-packages/s1denoise-0.1-py3.7.egg/s1denoise/denoising_parameters_S1A.npz'

# Mode
mode = sys.argv[2]

# GRD Mode
grd_mode = sys.argv[3]

# Polarization file prefix
pol_mode = sys.argv[4]

# dir path to noise scaling training data
in_path = sys.argv[5]

# dir to save updated output file with coefficients
out_path = sys.argv[6]

# flag to update npz file with coefficients
update_npz_files = True

# dicts with sub-swaths number and polarization
swaths_number = {'IW': 3, 'EW': 5}[mode]
swath_names = ['%s%s' % (mode,iSW) for iSW in range(1,swaths_number+1)]
polarisation = {'1SDH':'HV', '1SDV':'VH'}[pol_mode]

if not platform in ['S1A', 'S1B']:
    print('The input data must be S1A or S1B')
    exit()

if not mode in ['EW', 'IW']:
    print('The mode of the input data must be EW or IW')
    exit()

if not grd_mode in ['GRDM', 'GRDH']:
    print('The mode of the input GRD data must be GRDM or GRDH')
    exit()

if update_npz_files:
    outfile_npz_file = '%s/ns_%s_%s_denoising_parameters_%s.npz' % (out_path, mode, grd_mode, platform)

npzFilesAll = sorted(glob.glob('%s/%s_%s_%s_%s_*_noiseScaling.npz' % (in_path, platform,
                                                                        mode, grd_mode, pol_mode)))

# Check quality disclaimer #30 and #31 in https://qc.sentinel1.eo.esa.int/disclaimer/

npzFiles = []
for li, npzFile in enumerate(npzFilesAll):
    startDateTime = datetime.datetime.strptime(os.path.basename(npzFile).split('/')[-1][17:32], "%Y%m%dT%H%M%S")
    endDateTime = datetime.datetime.strptime(os.path.basename(npzFile).split('/')[-1][33:48], "%Y%m%dT%H%M%S")
    if (     platform=='S1A'
         and startDateTime >= datetime.datetime(2018,3,13,1,0,42)
         and endDateTime <= datetime.datetime(2018,3,15,14,1,26) ):
        continue
    elif (     platform=='S1B'
           and startDateTime >= datetime.datetime(2018,3,13,2,43,5)
           and endDateTime <= datetime.datetime(2018,3,15,15,19,30) ):
        continue
    else:
        npzFiles.append(npzFile)

# stack processed files
IPFversion = {'%s' % li: [] for li in swath_names}
powerDifference = {'%s' % li: [] for li in swath_names}
scalingFactor = {'%s' % li: [] for li in swath_names}
correlationCoefficient = {'%s' % li: [] for li in swath_names}
fitResidual = {'%s' % li: [] for li in swath_names}
acqDate = {'%s' % li: [] for li in swath_names}

for npzFile in npzFiles:
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    npz.allow_pickle=True

    for iSW in swath_names:
        numberOfSubblocks = np.unique([
            len(npz[iSW].item()[key])
            for key in ['scalingFactor', 'correlationCoefficient', 'fitResidual'] ])
        if numberOfSubblocks.size != 1:
            print('*** numberOfSubblocks are not consistent for all estimations.')
            continue
        numberOfSubblocks = numberOfSubblocks.item()
        powerDifference[iSW].append([
              np.nanmean(10*np.log10(npz[iSW].item()['sigma0'][li]))
            - np.nanmean(10*np.log10(npz[iSW].item()['noiseEquivalentSigma0'][li]))
            for li in range(numberOfSubblocks) ])
        scalingFactor[iSW].append(npz[iSW].item()['scalingFactor'])
        correlationCoefficient[iSW].append(npz[iSW].item()['correlationCoefficient'])
        fitResidual[iSW].append(npz[iSW].item()['fitResidual'])
        dummy = [ IPFversion[iSW].append(npz['IPFversion']) for li in range(numberOfSubblocks)]
        dummy = [ acqDate[iSW].append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))
                  for li in range(numberOfSubblocks) ]

# compute fit values
noiseScalingParameters = {'%s' % li: {} for li in swath_names}
noiseScalingParametersRMSE = {'%s' % li: {} for li in swath_names}

for IPFv in np.arange(2.4, 4.0, 0.1):
    for iSW in swath_names:
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['%s%s' % (mode, iSW)])==2.72,
                                   np.array(acqDate['%s%s' % (mode, iSW)]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion[iSW])*10)/10.), IPFv, atol=0.01)
        if valid.sum()==0:
            continue

        pd = np.hstack(powerDifference[iSW])[valid]
        sf = np.hstack(scalingFactor[iSW])[valid]
        cc = np.hstack(correlationCoefficient[iSW])[valid]
        fr = np.hstack(fitResidual[iSW])[valid]

        # weight for fitting: higher weights for high correlation and low RMSE from K-fitting
        w = cc / fr
        # VERY COMPLEX: fitting of K to powerDifference with degree=0
        # Here we find optimal value of K (just one value since degree of fitted polynom is 0).
        # That optimal value corresponds to:
        #  * high density of powerDifference values: This high density appears where powerDifference
        #    is low. I.e. where signal is low (low wind conditions).
        #  * high weights: where correlation is high and rmse is low
        # y using this fitting we avoid neccesity to choose scenes with low wind manualy.
        fitResults = np.polyfit(pd, sf, deg=0, w=w)

        # Results
        noiseScalingParameters[iSW]['%.1f' % IPFv] = fitResults[0]
        noiseScalingParametersRMSE[iSW]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-sf)**2 * w) / np.sum(w))

    # Plot data distribution
    for iSW in swath_names:
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion[iSW])==2.72,
                                   np.array(acqDate[iSW]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion[iSW])*10)/10.), IPFv, atol=0.01)

        if valid.sum()==0:
            continue

        pd = np.hstack(powerDifference[iSW])[valid]
        print(pd)
        sf = np.hstack(scalingFactor[iSW])[valid]
        cc = np.hstack(correlationCoefficient[iSW])[valid]
        fr = np.hstack(fitResidual[iSW])[valid]
        w = cc / fr

        fitResults = np.polyfit(pd, sf, deg=0, w=w)
        print(fitResults[0])

if update_npz_files:
    print('\ngoing to update coefficients for the noise scaling...')
    data = np.load(path_to_coefficients_npz)
    data.allow_pickle = True

    # Restore dictonaries for the data
    d_s1 = {key: data[key].item() for key in data}

    print('\nnew obtained coefficients')

    # loop over each mode and each IPF
    for ss in swath_names:
        for item in noiseScalingParameters[ss].items():
            ipf_ver = item[0]
            print(ipf_ver)

            # try replace existing value
            try:
                d_s1['%s' % polarisation]['noiseScalingParameters'][ss][ipf_ver] = \
                    noiseScalingParameters[ss][ipf_ver]
                print('success adding new record %s (IPF: %s)...' % (ss, ipf_ver))
            except:
                # make a new record
                print('trying adding new record %s (IPF: %s)...' % (ss, ipf_ver))
                d_s1['%s' % polarisation]['noiseScalingParameters'].update(
                    {ss: {ipf_ver: noiseScalingParameters[ss][ipf_ver]}}
                )

print('\nPrinting updated coefficients for double check:')
for ss in swath_names:
    for item in noiseScalingParameters[ss].items():
        ipf_ver = item[0]
        print('\nMode: %s, IPF: %s, Value: %s' % (ss, ipf_ver, d_s1[polarisation]['noiseScalingParameters'][ss][ipf_ver]))

# save updated version
np.savez(outfile_npz_file, **d_s1)
print('\nDone!\n')

