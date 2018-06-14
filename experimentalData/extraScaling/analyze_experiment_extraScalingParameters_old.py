import os, sys, glob, datetime
import numpy as np
from scipy.optimize import curve_fit


platform = 'S1A'
#platform = 'S1B'

# import data
npzFilesAll = sorted(glob.glob('%s_EW_GRDM_1SDH_*_extraScaling.npz' % platform))
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
extraScalingFactor = {'EW%s' % li:[] for li in range(1,6)}
signalPlusNoiseToNoiseRatio = {'EW%s' % li:[] for li in range(1,6)}
noiseNormalizedStandardDeviation = {'EW%s' % li:[] for li in range(1,6)}
acqDate = {'EW%s' % li: [] for li in range(1,6)}
for li, npzFile in enumerate(npzFiles):
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    for iSW in range(1,6):
        npzSW = npz['EW%s' % iSW].item()
        extraScalingFactor['EW%s' % iSW].append(npzSW['extraScalingFactor'])
        signalPlusNoiseToNoiseRatio['EW%s' % iSW].append(npzSW['signalPlusNoiseToNoiseRatio'])
        noiseNormalizedStandardDeviation['EW%s' % iSW].append(npzSW['noiseNormalizedStandardDeviation'])
        IPFversion['EW%s' % iSW].append(npz['IPFversion'])
        acqDate['EW%s' % iSW].append(datetime.datetime.strptime(npzFile.split('_')[4], '%Y%m%dT%H%M%S'))


def model_function(x,a,k,b,v,q,c):
    return a+( (k-a) / (c+q*np.exp(-b*x))**(1/v) )
#def model_function(x,a,b,c,d):
#    return a * b**((x+c)/d) + 1


# compute mean values
extraScalingParameters = {'EW%s' % li: {} for li in range(1,6)}
extraScalingParameters['SNNR'] = np.linspace(-25,+25,501)
for IPFv in np.arange(2.4, 4.0, 0.1):
    for iSW in range(1,6):
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['EW%s' % iSW])==2.72,
                                   np.array(acqDate['EW%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion['EW%s' % iSW])*10)/10.), IPFv, atol=0.01)
        if valid.sum()==0:
            continue
        # extra scaling parameters
        snnr = 10*np.log10(np.hstack(np.array(signalPlusNoiseToNoiseRatio['EW%s' % iSW])[valid]))
        esf = np.hstack(np.array(extraScalingFactor['EW%s' % iSW])[valid])
        esf[snnr>=3] = 1.0
        popt, pcov = curve_fit( model_function, snnr, esf,
                                p0=[100,1,1,1,0.005,1], maxfev=10000 )
        fittedCurve = model_function(extraScalingParameters['SNNR'] , *popt)
        extraScalingParameters['EW%s' % iSW]['%.1f' % IPFv] = fittedCurve - fittedCurve.min() + 1.


# compute mean values
noiseVarianceParameters = {'EW%s' % li: {} for li in range(1,6)}
for IPFv in np.arange(2.4, 4.0, 0.1):
    for iSW in range(1,6):
        if IPFv==2.7 and platform=='S1B':
            valid = np.logical_and(np.array(IPFversion['EW%s' % iSW])==2.72,
                                   np.array(acqDate['EW%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
        else:
            valid = np.isclose((np.trunc(np.array(IPFversion['EW%s' % iSW])*10)/10.), IPFv, atol=0.01)
        if valid.sum()==0:
            continue
        # noise variation parameters
        snnr = np.hstack(np.array(signalPlusNoiseToNoiseRatio['EW%s' % iSW])[valid])
        nnsd = np.hstack(np.array(noiseNormalizedStandardDeviation['EW%s' % iSW])[valid])
        fitMask = np.logical_and(snnr >= 1.0,
                                 snnr <= np.percentile(snnr, 95),
                                 nnsd <= np.percentile(nnsd, 95))
        noiseVarianceParameters['EW%s' % iSW]['%.1f' % IPFv] = np.polyval(
            np.polyfit(snnr[fitMask], nnsd[fitMask], w=1./snnr[fitMask], deg=1), 1.0)
            

for iSW in range(1,6):
    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and(np.array(IPFversion['EW%s' % iSW])==2.72,
                               np.array(acqDate['EW%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(np.array(IPFversion['EW%s' % iSW])*10)/10.), IPFv, atol=0.01)
    if valid.sum()==0:
        continue
    # noise variation parameters
    snnr = 10*np.log10(np.hstack(np.array(signalPlusNoiseToNoiseRatio['EW%s' % iSW])[valid]))
    esf = np.hstack(np.array(extraScalingFactor['EW%s' % iSW])[valid])
    esf[snnr>=3] = 1.0
    plt.subplot(1,5,iSW); plt.hold(0)
    plt.hist2d(snnr, esf, bins=251, range=[[-5,+5],[0,10]], cmin=1)
plt.tight_layout()


for iSW in range(1,6):
    if IPFv==2.7 and platform=='S1B':
        valid = np.logical_and(np.array(IPFversion['EW%s' % iSW])==2.72,
                               np.array(acqDate['EW%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
    else:
        valid = np.isclose((np.trunc(np.array(IPFversion['EW%s' % iSW])*10)/10.), IPFv, atol=0.01)
    if valid.sum()==0:
        continue
    # noise variation parameters
    snnr = np.hstack(np.array(signalPlusNoiseToNoiseRatio['EW%s' % iSW])[valid])
    nnsd = np.hstack(np.array(noiseNormalizedStandardDeviation['EW%s' % iSW])[valid])
    fitMask = np.logical_and(snnr <= np.percentile(snnr, 95), nnsd <= np.percentile(nnsd, 95))
    plt.subplot(1,5,iSW); plt.hold(0)
    plt.hist2d(snnr, nnsd, bins=501, range=[[0,10],[0,5]], cmin=1)
    plt.hold(1), plt.plot([1,1],[0,5], linewidth=0.5, color='r')
plt.tight_layout()

