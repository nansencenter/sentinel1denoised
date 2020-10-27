import os
import sys
import glob
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sys import exit

# run example:
# run analyze_experiment_powerBalancingParameters.py S1A  /mnt/sverdrup-2/sat_auxdata/denoise/dolldrums/zip

# Instrument
platform = sys.argv[1]

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
    path_to_trained_npz = '/Home/denemc/miniconda3/envs/py3s1denoise/lib/python3.7/site-packages/s1denoise-0.1-py3.7.egg/s1denoise/denoising_parameters_%s.npz' % platform
    outfile_npz_file = '%s/ns_%s_%s_denoising_parameters_%s.npz' % (out_path, mode, grd_mode, platform)

npzFilesAll = sorted(glob.glob('%s/%s_%s_%s_%s_*_powerBalancing.npz' % (in_path, platform,
                                                                        mode, grd_mode, pol_mode)))

###################

plt.clf()
plt.figure(figsize=(15,4))

# update npz files
update_npz_files = True

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

    numberOfSubblocks = np.unique([ len(npz['%s%s' % (mode,iSW)].item()['balancingPower'])
                                    for iSW in range(1,{'IW':4, 'EW':6}[mode]) ])
    if numberOfSubblocks.size != 1:
        print('*** numberOfSubblocks are not consistent for all subswaths.')
        continue
    numberOfSubblocks = numberOfSubblocks.item()

    for li in range(numberOfSubblocks):
        powerDifference.append([
              np.nanmean(10*np.log10(npz['%s%s' % (mode,iSW)].item()['sigma0'][li]))
            - np.nanmean(10*np.log10(npz['%s%s' % (mode,iSW)].item()['noiseEquivalentSigma0'][li]))
            for iSW in range(1,{'IW':4, 'EW':6}[mode]) ])
        balancingPower.append([
            npz['%s%s' % (mode,iSW)].item()['balancingPower'][li]
            for iSW in range(1,{'IW':4, 'EW':6}[mode]) ])
        correlationCoefficient.append([
            npz['%s%s' % (mode,iSW)].item()['correlationCoefficient'][li]
            for iSW in range(1,{'IW':4, 'EW':6}[mode]) ])
        fitResidual.append([
            npz['%s%s' % (mode,iSW)].item()['fitResidual'][li]
            for iSW in range(1,{'IW':4, 'EW':6}[mode]) ])
        IPFversion.append(npz['IPFversion'])
        acqDate.append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))

powerDifference = np.array(powerDifference)
balancingPower = np.array(balancingPower)
correlationCoefficient = np.array(correlationCoefficient)
fitResidual = np.array(fitResidual)
IPFversion = np.array(IPFversion)
acqDate = np.array(acqDate)

# compute fit values
powerBalancingParameters = {'%s%s' % (mode,li): {} for li in range(1, {'IW':4, 'EW':6}[mode])}
powerBalancingParametersRMSE = {'%s%s' % (mode,li): {} for li in range(1, {'IW':4, 'EW':6}[mode])}

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

    for iSW in range(1,{'IW':4, 'EW':6}[mode]):
        bp = balancingPower[valid][:,iSW-1]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)
        powerBalancingParameters['%s%s' % (mode,iSW)]['%.1f' % IPFv] = fitResults[0]
        powerBalancingParametersRMSE['%s%s' % (mode,iSW)]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-bp)**2 * w) / np.sum(w))

    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and( IPFversion==2.72,
                                acqDate < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(IPFversion*10)/10.), IPFv, atol=0.01)

    pd = np.mean(powerDifference[valid], axis=1)
    cc = np.min(correlationCoefficient[valid], axis=1)
    fr = np.max(fitResidual[valid], axis=1)
    w = cc / fr

    for iSW in range(1,{'IW':4, 'EW':6}[mode]):
        bp = balancingPower[valid][:,iSW-1]
        fitResults = np.polyfit(pd, bp, deg=0, w=w)
        plt.subplot(1,({'IW':4, 'EW':6}[mode])-1,iSW); plt.hold(0)
        plt.hist2d(bp,pd,bins=100,cmin=1,range=[[-2e-3,+2e-3],[-5,15]])
        plt.hold(1)
        plt.plot(np.polyval(fitResults, np.linspace(-5,+15,2)), np.linspace(-5,+15,2), linewidth=0.5, color='r')
        plt.plot([-2e-3,+2e-3],[0,0], linewidth=0.5, color='k')

# Save a figure with statistics on noise scaling
plt.tight_layout()
plt.savefig('%s_%s_%s_power_balancing.png' % (platform, mode), bbox_inches='tight', dpi=600)

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

    # Loop over values for each mode and each IPF
    for i in range(1, num_ss[mode] + 1):
        ss = '%s%d' % (mode, i)

        for item in powerBalancingParameters[ss].items():
            ipf_ver = item[0]

            # try replace existing value
            try:
                d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['powerBalancingParameters'][ss][ipf_ver] = \
                    powerBalancingParameters[ss][ipf_ver]
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
            except:
                # make a new record
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
                d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['powerBalancingParameters'].update(
                    {ss: {ipf_ver: powerBalancingParameters[ss][ipf_ver]}}
                )

print('\nPrinting updated coefficients for double check:')

for i in range(1, num_ss[mode] + 1):
    ss = '%s%d' % (mode, i)
    for item in powerBalancingParameters[ss].items():
        ipf_ver = item[0]
        print('\nMode: %s, IPF: %s, Value: %s' % (ss, ipf_ver, d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['powerBalancingParameters'][ss][ipf_ver]))

# save updated version
np.savez(outfile_npz_file, **d_s1)
print('\nDone!\n')