# -*- coding: utf-8 -*-
# Modified: 2020-06-20
# by DD

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(15,4))

platform = 'S1A'
#platform = 'S1B'
mode = 'IW'
grd_mode = 'GRDH'
region = 'NA'

# dir path to noise scaling training data
in_path = sys.argv[1]
out_path = sys.argv[2]

# update npz files
update_npz_files = True

# update npz files
update_npz_files = True
print('Update = %s' % update_npz_files)

if update_npz_files:
    path_to_trained_npz = '/Home/denemc/miniconda3/envs/py3s1denoise/lib/python3.7/site-packages/s1denoise-0.1-py3.7.egg/s1denoise/denoising_parameters_%s.npz' % platform
    outfile_npz_file = '%s/pb_%s_%s_denoising_parameters_%s.npz' % (out_path, mode, grd_mode, platform)

# scan for PB npz files
npzFilesAll = sorted(glob.glob('%s/%s_%s_%s_1SDH_*_powerBalancing.npz' % (in_path, platform,
                                                                        mode, grd_mode)))

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

'''
if mode == 'EW':
    n = 6
    IPFversion = {'EW%s' % li: [] for li in range(1,6)}
    powerDifference = {'EW%s' % li: [] for li in range(1,6)}
    balancingPower = {'EW%s' % li: [] for li in range(1,6)}
    correlationCoefficient = {'EW%s' % li: [] for li in range(1,6)}
    fitResidual = {'EW%s' % li: [] for li in range(1,6)}
    acqDate = {'EW%s' % li: [] for li in range(1,6)}

if mode == 'IW':
    n = 4
    IPFversion = {'IW%s' % li: [] for li in range(1, 4)}
    powerDifference = {'IW%s' % li: [] for li in range(1, 4)}
    balancingPower = {'IW%s' % li: [] for li in range(1, 4)}
    correlationCoefficient = {'IW%s' % li: [] for li in range(1, 4)}
    fitResidual = {'IW%s' % li: [] for li in range(1, 4)}
    acqDate = {'IW%s' % li: [] for li in range(1, 4)}
'''

if mode == 'EW':
    n = 6
if mode == 'IW':
    n = 4

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

    if mode == 'EW':
        numberOfSubblocks = np.unique([ len(npz['EW%s' % iSW].item()['balancingPower'])
                                        for iSW in range(1,n) ])
        if numberOfSubblocks.size != 1:
            print('*** numberOfSubblocks are not consistent for all subswaths.')
            continue
        numberOfSubblocks = numberOfSubblocks.item()

        for li in range(numberOfSubblocks):
            powerDifference.append([
                  np.nanmean(10*np.log10(npz['EW%s' % iSW].item()['sigma0'][li]))
                - np.nanmean(10*np.log10(npz['EW%s' % iSW].item()['noiseEquivalentSigma0'][li]))
                for iSW in range(1,n) ])
            balancingPower.append([
                npz['EW%s' % iSW].item()['balancingPower'][li]
                for iSW in range(1,n) ])
            correlationCoefficient.append([
                npz['EW%s' % iSW].item()['correlationCoefficient'][li]
                for iSW in range(1,n) ])
            fitResidual.append([
                npz['EW%s' % iSW].item()['fitResidual'][li]
                for iSW in range(1,n) ])
            IPFversion.append(npz['IPFversion'])
            acqDate.append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))

    if mode == 'IW':
        numberOfSubblocks = np.unique([len(npz['IW%s' % iSW].item()['balancingPower'])
                                       for iSW in range(1,n)])
        if numberOfSubblocks.size != 1:
            print('*** numberOfSubblocks are not consistent for all subswaths.')
            continue
        numberOfSubblocks = numberOfSubblocks.item()
        for li in range(numberOfSubblocks):
            powerDifference.append([
                np.nanmean(10 * np.log10(npz['IW%s' % iSW].item()['sigma0'][li]))
                - np.nanmean(10 * np.log10(npz['IW%s' % iSW].item()['noiseEquivalentSigma0'][li]))
                for iSW in range(1,n)])
            balancingPower.append([
                npz['IW%s' % iSW].item()['balancingPower'][li]
                for iSW in range(1,n)])
            correlationCoefficient.append([
                npz['IW%s' % iSW].item()['correlationCoefficient'][li]
                for iSW in range(1,n)])
            fitResidual.append([
                npz['IW%s' % iSW].item()['fitResidual'][li]
                for iSW in range(1,n)])
            IPFversion.append(npz['IPFversion'])
            acqDate.append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))


powerDifference = np.array(powerDifference)
balancingPower = np.array(balancingPower)
correlationCoefficient = np.array(correlationCoefficient)
fitResidual = np.array(fitResidual)
IPFversion = np.array(IPFversion)
acqDate = np.array(acqDate)

# compute fit values

if mode == 'EW':
    powerBalancingParameters = {'EW%s' % li: {} for li in range(1, n)}
    powerBalancingParametersRMSE = {'EW%s' % li: {} for li in range(1, n)}
if mode == 'IW':
    powerBalancingParameters = {'IW%s' % li: {} for li in range(1, n)}
    powerBalancingParametersRMSE = {'IW%s' % li: {} for li in range(1, n)}

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

    for iSW in range(1,n):
        bp = balancingPower[valid][:,iSW-1]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)

        if mode == 'EW':
            powerBalancingParameters['EW%s' % iSW]['%.1f' % IPFv] = fitResults[0]
            powerBalancingParametersRMSE['EW%s' % iSW]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-bp)**2 * w) / np.sum(w))

        if mode == 'IW':
            powerBalancingParameters['IW%s' % iSW]['%.1f' % IPFv] = fitResults[0]
            powerBalancingParametersRMSE['IW%s' % iSW]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-bp)**2 * w) / np.sum(w))

    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and( IPFversion==2.72,
                                acqDate < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(IPFversion*10)/10.), IPFv, atol=0.01)

    pd = np.mean(powerDifference[valid], axis=1)
    cc = np.min(correlationCoefficient[valid], axis=1)
    fr = np.max(fitResidual[valid], axis=1)
    w = cc / fr

    for iSW in range(1,n):
        bp = balancingPower[valid][:,iSW-1]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)
        plt.subplot(1,n-1,iSW); plt.hold(0)
        plt.hist2d(bp,pd,bins=100,cmin=1,range=[[-2e-3,+2e-3],[-5,15]])
        plt.hold(1)
        plt.plot(np.polyval(fitResults, np.linspace(-5,+15,2)), np.linspace(-5,+15,2), linewidth=0.5, color='r')
        plt.plot([-2e-3,+2e-3],[0,0], linewidth=0.5, color='k')

# Save a figure with statistics on noise scaling
plt.tight_layout()
plt.savefig('%s_%s_%s_power_balancing.png' % (platform, mode, region), bbox_inches='tight', dpi=600)

# if update_npz_files
num_ss = {'EW': 5, 'IW': 3}

if update_npz_files:
    print('\ngoing to update coeffients for the power balancing...')
    data = np.load(path_to_trained_npz)
    data.allow_pickle = True

    # Restore dictonaries for the data
    d_s1 = {key: data[key].item() for key in data}

    '''
    print('\nold coefficients:')
    try:
        for i in range(1,num_ss[mode]+1):
            ss = '%s%d' % (mode, i)
            print(ss, d_s1['HV']['noiseScalingParameters'][ss][str(npz['IPFversion'])])
    except:
        print('\nNo old coefficients\n')
    '''

    print('\nnew obtained coefficients')

    # Go through values for each mode and each IPF

    for i in range(1, num_ss[mode] + 1):
        ss = '%s%d' % (mode, i)

        for item in powerBalancingParameters[ss].items():
            ipf_ver = item[0]

            # try replace existing value
            try:
                d_s1['HV']['powerBalancingParameters'][ss][ipf_ver] = \
                    powerBalancingParameters[ss][ipf_ver]
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
            except:
                # make a new record
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
                d_s1['HV']['powerBalancingParameters'].update(
                    {ss: {ipf_ver: powerBalancingParameters[ss][ipf_ver]}}
                )

print('\nPrinting updated coefficients for double check:')

for i in range(1, num_ss[mode] + 1):
    ss = '%s%d' % (mode, i)
    for item in powerBalancingParameters[ss].items():
        ipf_ver = item[0]
        print('\nMode: %s, IPF: %s, Value: %s' % (ss, ipf_ver, d_s1['HV']['powerBalancingParameters'][ss][ipf_ver]))

# save updated version
np.savez(outfile_npz_file, **d_s1)
print('\nDone!\n')



''' !TODO: delete OLD
if update_npz_files:
    print('\ngoing to update coeffients for the power balancing...')
    data = np.load(path_to_trained_npz)
    data.allow_pickle = True

    # Restore dictonaries for the data
    d_s1 = {key: data[key].item() for key in data}

    print('\nold coefficients:')
    for i in range(1,num_ss[mode]+1):
        ss = '%s%d' % (mode, i)
        print(ss, d_s1['HV']['powerBalancingParameters'][ss][str(npz['IPFversion'])])

    print('\nnew obtained coefficients')
    for i in range(1, num_ss[mode] + 1):
        ss = '%s%d' % (mode, i)
        print(ss, powerBalancingParameters[ss][str(npz['IPFversion'])])

        # replace values
        d_s1['HV']['powerBalancingParameters'][ss][str(npz['IPFversion'])] = \
        powerBalancingParameters[ss][str(npz['IPFversion'])]

    print('\nupdated coefficients:')
    for i in range(1, num_ss[mode] + 1):
        ss = '%s%d' % (mode, i)
        print(ss, d_s1['HV']['powerBalancingParameters'][ss][str(npz['IPFversion'])])

    # save updated version
    np.savez(outfile_npz_file, **d_s1)
    print('\ndone!\n')
'''
