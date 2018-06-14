import os, sys, glob, datetime
import numpy as np

platform = 'S1A'
#platform = 'S1B'

# import data
npzFilesAll = sorted(glob.glob('%s_EW_GRDM_1SDH_*_noiseScaling.npz' % platform))
# Check quality disclaimer #30 and #31 in https://qc.sentinel1.eo.esa.int/disclaimer/
npzFiles = []
for li, npzFile in enumerate(npzFilesAll):
    startDateTime = datetime.datetime.strptime(npzFile.split('/')[-1][17:32], "%Y%m%dT%H%M%S")
    endDateTime = datetime.datetime.strptime(npzFile.split('/')[-1][33:48], "%Y%m%dT%H%M%S")
    if (     platform=='S1A'
         and startDateTime >= datetime.datetime(2018,03,13,01,00,42)
         and endDateTime <= datetime.datetime(2018,03,15,12,00,00) ):
        continue
    elif (     platform=='S1B'
           and startDateTime >= datetime.datetime(2018,03,13,02,43,05)
           and endDateTime <= datetime.datetime(2018,03,15,12,00,00) ):
        continue
    else:
        npzFiles.append(npzFile)
IPFversion = {'EW%s' % li: [] for li in range(1,6)}
powerDifference = {'EW%s' % li: [] for li in range(1,6)}
scalingFactor = {'EW%s' % li: [] for li in range(1,6)}
correlationCoefficient = {'EW%s' % li: [] for li in range(1,6)}
fitResidual = {'EW%s' % li: [] for li in range(1,6)}
acqDate = {'EW%s' % li: [] for li in range(1,6)}
for npzFile in npzFiles:
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    for iSW in range(1,6):
        numberOfSubblocks = np.unique([
            len(npz['EW%s' % iSW].item()[key])
            for key in ['scalingFactor', 'correlationCoefficient', 'fitResidual'] ])
        if numberOfSubblocks.size != 1:
            print('*** numberOfSubblocks are not consistent for all estimations.')
            continue
        powerDifference['EW%s' % iSW].append([
              np.nanmean(10*np.log10(npz['EW%s' % iSW].item()['sigma0'][li]))
            - np.nanmean(10*np.log10(npz['EW%s' % iSW].item()['noiseEquivalentSigma0'][li]))
            for li in range(numberOfSubblocks) ])
        scalingFactor['EW%s' % iSW].append(npz['EW%s' % iSW].item()['scalingFactor'])
        correlationCoefficient['EW%s' % iSW].append(npz['EW%s' % iSW].item()['correlationCoefficient'])
        fitResidual['EW%s' % iSW].append(npz['EW%s' % iSW].item()['fitResidual'])
        dummy = [ IPFversion['EW%s' % iSW].append(npz['IPFversion']) for li in range(numberOfSubblocks)]
        dummy = [ acqDate['EW%s' % iSW].append(datetime.datetime.strptime(npzFile.split('_')[4], '%Y%m%dT%H%M%S'))
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

# compute fit values
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
        w = cc / fr
        fitResults = np.polyfit(pd, sf, deg=0, w=w)
        noiseScalingParameters['EW%s' % iSW]['%.1f' % IPFv] = fitResults[0]
        noiseScalingParametersRMSE['EW%s' % iSW]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-sf)**2 * w) / np.sum(w))

'''
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
    w = cc / fr
    fitResults = np.polyfit(pd, sf, deg=0, w=w)
    print fitResults[0]
    plt.subplot(1,5,iSW); plt.hold(0)
    plt.hist2d(sf,pd,bins=100,cmin=1,range=[[0,3],[-5,15]])
    plt.hold(1)
    plt.plot(np.polyval(fitResults, np.linspace(-5,+15,2)), np.linspace(-5,+15,2), linewidth=0.5, color='r')
    plt.plot([0,3],[0,0], linewidth=0.5, color='k')
plt.tight_layout()
'''

