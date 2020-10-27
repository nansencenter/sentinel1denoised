import os
import sys
import glob
import datetime
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Mode
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
IPFversion = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}
powerDifference = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}
scalingFactor = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}
correlationCoefficient = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}
fitResidual = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}
acqDate = {'%s%s' % (mode,li): [] for li in range(1,{'IW':4, 'EW':6}[mode])}

for npzFile in npzFiles:
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    npz.allow_pickle=True

    for iSW in range(1,{'IW':4, 'EW':6}[mode]):
        numberOfSubblocks = np.unique([
            len(npz['%s%s' % (mode,iSW)].item()[key])
            for key in ['scalingFactor', 'correlationCoefficient', 'fitResidual'] ])
        if numberOfSubblocks.size != 1:
            print('*** numberOfSubblocks are not consistent for all estimations.')
            continue
        numberOfSubblocks = numberOfSubblocks.item()
        powerDifference['%s%s' % (mode,iSW)].append([
              np.nanmean(10*np.log10(npz['%s%s' % (mode,iSW)].item()['sigma0'][li]))
            - np.nanmean(10*np.log10(npz['%s%s' % (mode,iSW)].item()['noiseEquivalentSigma0'][li]))
            for li in range(numberOfSubblocks) ])
        scalingFactor['%s%s' % (mode,iSW)].append(npz['%s%s' % (mode,iSW)].item()['scalingFactor'])
        correlationCoefficient['%s%s' % (mode,iSW)].append(npz['%s%s' % (mode,iSW)].item()['correlationCoefficient'])
        fitResidual['%s%s' % (mode,iSW)].append(npz['%s%s' % (mode,iSW)].item()['fitResidual'])
        dummy = [ IPFversion['%s%s' % (mode,iSW)].append(npz['IPFversion']) for li in range(numberOfSubblocks)]
        dummy = [ acqDate['%s%s' % (mode,iSW)].append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))
                  for li in range(numberOfSubblocks) ]

'''
# compute mean values
thres = 20
noiseScalingParameters = {'EW%s' % li: {} for li in range(1,6)}
noiseScalingParametersRMSE = {'EW%s' % li: {} for li in range(1,6)}
for IPFv in np.arange(2.4, 4.0, 0.1):
    for iSW in range(1,6):
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['EW%s' % iSW])==2.72,
                                   np.array(acqDate['EW%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion['EW%s' % iSW])*10)/10.), IPFv, atol=0.01)
        if valid.sum()==0:
            continue
        pd = np.hstack(powerDifference['EW%s' % iSW])[valid]
        sf = np.hstack(scalingFactor['EW%s' % iSW])[valid]
        cc = np.hstack(correlationCoefficient['EW%s' % iSW])[valid]
        fr = np.hstack(fitResidual['EW%s' % iSW])[valid]
        #goodSamples = (pd <= 2) * (cc >= 0.9) * (fr <= 5e-16)
        goodSamples = (   (pd <= np.percentile(pd[np.isfinite(pd)],thres))
                        * (cc >= np.percentile(cc[np.isfinite(cc)],100-thres))
                        * (fr <= np.percentile(fr[np.isfinite(fr)],thres)) )
        print('%s: %d / %d' % (IPFv, goodSamples.sum(), len(goodSamples)))
        noiseScalingParameters['EW%s' % iSW]['%.1f' % IPFv] = np.nanmean(sf[goodSamples])
        noiseScalingParametersRMSE['EW%s' % iSW]['%.1f' % IPFv] = np.sqrt(np.sum((np.nanmean(sf[goodSamples])-sf[goodSamples])**2) / goodSamples.sum())
'''

plt.clf()
plt.figure(figsize=(15,4))

# compute fit values
noiseScalingParameters = {'%s%s' % (mode, li): {} for li in range(1,{'IW':4, 'EW':6}[mode])}
noiseScalingParametersRMSE = {'%s%s' % (mode, li): {} for li in range(1,{'IW':4, 'EW':6}[mode])}

for IPFv in np.arange(2.4, 4.0, 0.1):
    for iSW in range(1,{'IW':4, 'EW':6}[s1.obsMode]):
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['%s%s' % (mode, iSW)])==2.72,
                                   np.array(acqDate['%s%s' % (mode, iSW)]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion['%s%s' % (mode, iSW)])*10)/10.), IPFv, atol=0.01)
        if valid.sum()==0:
            continue

        pd = np.hstack(powerDifference['%s%s' % (mode,iSW)])[valid]
        sf = np.hstack(scalingFactor['%s%s' % (mode,iSW)])[valid]
        cc = np.hstack(correlationCoefficient['%s%s' % (mode,iSW)])[valid]
        fr = np.hstack(fitResidual['%s%s' % (mode,iSW)])[valid]

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
        noiseScalingParameters['%s%s' % (mode,iSW)]['%.1f' % IPFv] = fitResults[0]
        noiseScalingParametersRMSE['%s%s' % (mode,iSW)]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-sf)**2 * w) / np.sum(w))

    # Plot data distribution
    for iSW in range(1,{'IW':4, 'EW':6}[mode]):
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['%s%s' % (mode, iSW)])==2.72,
                                   np.array(acqDate['%s%s' % (mode, iSW)]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion['%s%s' % (mode, iSW)])*10)/10.), IPFv, atol=0.01)

        if valid.sum()==0:
            continue

        pd = np.hstack(powerDifference['%s%s' % (mode, iSW)])[valid]
        print(pd)
        sf = np.hstack(scalingFactor['%s%s' % (mode, iSW)])[valid]
        cc = np.hstack(correlationCoefficient['%s%s' % (mode, iSW)])[valid]
        fr = np.hstack(fitResidual['%s%s' % (mode, iSW)])[valid]
        w = cc / fr

        fitResults = np.polyfit(pd, sf, deg=0, w=w)
        print(fitResults[0])
        plt.subplot(1,5,iSW); plt.hold(0)
        plt.hist2d(sf,pd,bins=100,cmin=1,range=[[0,3],[-5,15]])
        plt.hold(1)

        plt.plot(np.polyval(fitResults, np.linspace(-5,+15,2)), np.linspace(-5,+15,2), linewidth=0.5, color='r')
        plt.plot([0,3],[0,0], linewidth=0.5, color='k')

# Save a figure with statistics on noise scaling
plt.tight_layout()
plt.savefig('%s/%s_%s_scale_noise.png' % (out_path, platform, mode), bbox_inches='tight', dpi=600)

# if update_npz_files
num_ss = {'EW': 5, 'IW': 3}

if update_npz_files:
    print('\ngoing to update coefficients for the noise scaling...')
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

    # loop over each mode and each IPF
    for i in range(1, num_ss[mode] + 1):
        ss = '%s%d' % (mode, i)

        for item in noiseScalingParameters[ss].items():
            ipf_ver = item[0]

            # try replace existing value
            try:
                d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['noiseScalingParameters'][ss][ipf_ver] = \
                    noiseScalingParameters[ss][ipf_ver]
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
            except:
                # make a new record
                print('adding new record %s (IPF: %s)...' % (ss, ipf_ver))
                d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['noiseScalingParameters'].update(
                    {ss: {ipf_ver: noiseScalingParameters[ss][ipf_ver]}}
                )

print('\nPrinting updated coefficients for double check:')
for i in range(1, num_ss[mode] + 1):
    ss = '%s%d' % (mode, i)
    for item in noiseScalingParameters[ss].items():
        ipf_ver = item[0]
        print('\nMode: %s, IPF: %s, Value: %s' % (ss, ipf_ver, d_s1['%s' % {'1SDH':'HV', '1SDV':'VH'}[grd_mode]]['noiseScalingParameters'][ss][ipf_ver]))

# save updated version
np.savez(outfile_npz_file, **d_s1)
print('\nDone!\n')

