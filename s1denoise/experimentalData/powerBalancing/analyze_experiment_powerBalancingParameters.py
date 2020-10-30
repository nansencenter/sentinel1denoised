""" This python script process aggregated statistics
for individual npz files to obtain
final results in power balancing stage

run example:
python analyze_experiment_powerBalancingParameters.py S1A IW GRDH 1SDV /path/to/npz/files /out/path

Important note:
If you wish to generate updated npz file with coefficients you need to specify
a path to it in variable called 'path_to_trained_npz' and define 'update_npz_files' as True
"""

import os
import sys
import glob
import datetime
import numpy as np
from sys import exit

# Instrument
platform = sys.argv[1]

# flag to generate updated npz file with coefficients
update_npz_files = True

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

if not platform in ['S1A', 'S1B']:
    print('The input data must be S1A or S1B')
    exit()

if not mode in ['EW', 'IW']:
    print('The mode of the input data must be EW or IW')
    exit()

if not grd_mode in ['GRDM', 'GRDH']:
    print('The mode of the input GRD data must be GRDM or GRDH')
    exit()

# Save results as updated npz file with old and obtained coefficients
if update_npz_files:
    # Path to your npz file with coefficients
    path_to_trained_npz = '/Home/denemc/miniconda3/envs/py3s1denoise/lib/python3.7/site-packages/s1denoise-0.1-py3.7.egg/s1denoise/denoising_parameters_%s.npz' % platform
    # Path to resultant updated npz file with coefficients
    outfile_npz_file = '%s/ns_%s_%s_denoising_parameters_%s.npz' % (out_path, mode, grd_mode, platform)

npzFilesAll = sorted(glob.glob('%s/%s_%s_%s_%s_*_powerBalancing.npz' % (in_path, platform,
                                                                        mode, grd_mode, pol_mode)))

# dicts with sub-swaths number and polarization
swaths_number = {'IW': 3, 'EW': 5}[mode]
swath_names = ['%s%s' % (mode,iSW) for iSW in range(1,swaths_number+1)]
polarisation = {'1SDH':'HV', '1SDV':'VH'}[pol_mode]

# update npz files
update_npz_files = True

if update_npz_files:
    path_to_trained_npz = '/Home/denemc/miniconda3/envs/py3s1denoise/lib/python3.7/site-packages/s1denoise-0.1-py3.7.egg/s1denoise/denoising_parameters_%s.npz' % platform
    outfile_npz_file = '%s/pb_%s_%s_denoising_parameters_%s.npz' % (out_path, mode, grd_mode, platform)

# scan for PB npz files
npzFilesAll = sorted(glob.glob('%s/%s_%s_%s_*_powerBalancing.npz' % (in_path, platform,
                                                                        mode, grd_mode)))

# Check quality disclaimer #30 and #31 in https://qc.sentinel1.eo.esa.int/disclaimer/

npzFiles = []

for li, npzFile in enumerate(npzFilesAll):
    print(npzFile)
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
IPFversion = []
powerDifference = []
balancingPower = []
correlationCoefficient = []
fitResidual = []
acqDate = []

for npzFile in npzFiles:
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    npz.allow_pickle = True

    numberOfSubblocks = np.unique([ len(npz[iSW].item()['balancingPower'])
                                    for iSW in swath_names])
    if numberOfSubblocks.size != 1:
        print('*** numberOfSubblocks are not consistent for all subswaths.')
        continue
    numberOfSubblocks = numberOfSubblocks.item()

    for li in range(numberOfSubblocks):
        powerDifference.append([
              np.nanmean(10*np.log10(npz[iSW].item()['sigma0'][li]))
            - np.nanmean(10*np.log10(npz[iSW].item()['noiseEquivalentSigma0'][li]))
            for iSW in swath_names])
        balancingPower.append([
            npz[iSW].item()['balancingPower'][li]
            for iSW in swath_names])
        correlationCoefficient.append([
            npz[iSW].item()['correlationCoefficient'][li]
            for iSW in swath_names])
        fitResidual.append([
            npz[iSW].item()['fitResidual'][li]
            for iSW in swath_names])
        IPFversion.append(npz['IPFversion'])
        acqDate.append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))

powerDifference = np.array(powerDifference)
balancingPower = np.array(balancingPower)
correlationCoefficient = np.array(correlationCoefficient)
fitResidual = np.array(fitResidual)
IPFversion = np.array(IPFversion)
acqDate = np.array(acqDate)

# compute fit values
powerBalancingParameters = {li: {} for li in swath_names}
powerBalancingParametersRMSE = {li: {} for li in swath_names}

for IPFv in np.arange(2.4, 4.0, 0.1):
    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and( IPFversion==2.72,
                                acqDate < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(IPFversion*10)/10.), IPFv, atol=0.01)

    if valid.sum()==0:
        continue

    pd = np.mean(powerDifference[valid], axis=1)
    cc = np.min(correlationCoefficient[valid], axis=1)
    fr = np.max(fitResidual[valid], axis=1)
    w = cc / fr

    for iSW in range(0,swaths_number):
        bp = balancingPower[valid][:,iSW]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)
        powerBalancingParameters[swath_names[iSW]]['%.1f' % IPFv] = fitResults[0]
        powerBalancingParametersRMSE[swath_names[iSW]]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-bp)**2 * w) / np.sum(w))

    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and( IPFversion==2.72,
                                acqDate < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(IPFversion*10)/10.), IPFv, atol=0.01)

    pd = np.mean(powerDifference[valid], axis=1)
    cc = np.min(correlationCoefficient[valid], axis=1)
    fr = np.max(fitResidual[valid], axis=1)
    w = cc / fr

    for iSW in range(0,swaths_number):
        bp = balancingPower[valid][:,iSW]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)

# if update_npz_files
if update_npz_files:
    print('\nGoing to update power balancing coefficients...')
    data = np.load(path_to_trained_npz)
    data.allow_pickle = True

    # Restore dictonaries for the data
    d_s1 = {key: data[key].item() for key in data}

    pbname = 'powerBalancingParameters'

    # Create dict structure for coefficients if it does not exist
    for ss in swath_names:
        if polarisation not in d_s1:
            d_s1[polarisation] = {pbname: {}}
        if pbname not in d_s1[polarisation]:
            d_s1[polarisation][pbname] = {ss: {}}
        if ss not in d_s1[polarisation][pbname]:
            d_s1[polarisation][pbname][ss] = dict()

    # Loop over values for each mode and each IPF
    for ss in swath_names:
        for item in powerBalancingParameters[ss].items():
            ipf_ver = item[0]
            d_s1[polarisation]['powerBalancingParameters'][ss][ipf_ver] = powerBalancingParameters[ss][ipf_ver]
            print('\nAdding new record %s (IPF: %s)...' % (ss, ipf_ver))
            print('%s\n' % powerBalancingParameters[ss][ipf_ver])

print('\nPrinting updated coefficients for double check:')

for ss in swath_names:
    for item in powerBalancingParameters[ss].items():
        ipf_ver = item[0]
        print('\nMode: %s, IPF: %s, Value: %s' % (ss, ipf_ver, d_s1[polarisation]['powerBalancingParameters'][ss][ipf_ver]))

# save updated version
np.savez(outfile_npz_file, **d_s1)
print('\nDone!\n')