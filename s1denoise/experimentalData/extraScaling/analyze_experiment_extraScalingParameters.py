import os, sys, glob, datetime
import numpy as np
from scipy.optimize import curve_fit


platform = 'S1A'
#platform = 'S1B'

npzFilesAll = sorted(glob.glob('%s_EW_GRDM_1SDH_*_extraScaling.npz' % platform))
npzFiles = []
for li, npzFile in enumerate(npzFilesAll):
    startDateTime = datetime.datetime.strptime(npzFile.split('/')[-1][17:32], "%Y%m%dT%H%M%S")
    endDateTime = datetime.datetime.strptime(npzFile.split('/')[-1][33:48], "%Y%m%dT%H%M%S")
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
npz = np.load(npzFiles[0])
snnrEdges = npz['snnrEdges']
nnsdEdges = npz['nnsdEdges']
dBsnnrEdges = npz['dBsnnrEdges']
esfEdges = npz['esfEdges']
windowSizes = npz['windowSizes']
nBins = len(snnrEdges)-1
nnsdHist = {'EW%s' % li: np.zeros((len(windowSizes), nBins, nBins), dtype=np.int64)
            for li in range(1,6)}
esfHist = {'EW%s' % li: np.zeros((len(windowSizes), nBins, nBins), dtype=np.int64)
           for li in range(1,6)}
for li, npzFile in enumerate(npzFiles):
    sys.stdout.write('\rStacking ... %5d / %5d' % (li+1,len(npzFiles)))
    sys.stdout.flush()
    npz = np.load(npzFile)
    nnsdHistTmp = npz['noiseNormalizedStandardDeviationHistogram'].item()
    esfHistTmp = npz['extraScalingFactorHistogram'].item()
    for iSW in range(1,6):
        nnsdHist['EW%s' % iSW] += np.transpose(nnsdHistTmp['EW%s' % iSW], axes=(0,2,1))
        esfHist['EW%s' % iSW] += np.transpose(esfHistTmp['EW%s' % iSW], axes=(0,2,1))
np.savez_compressed(platform+'_extraScaling.npz', windowSizes=windowSizes,
                    snnrEdges=snnrEdges, nnsdEdges=nnsdEdges, nnsdHist=nnsdHist,
                    dBsnnrEdges=dBsnnrEdges, esfEdges=esfEdges, esfHist=esfHist)


npz = np.load(platform+'_extraScaling.npz')
snnrEdges = npz['snnrEdges']
nnsdEdges = npz['nnsdEdges']
windowSizes = npz['windowSizes']
nBins = len(snnrEdges)-1
fitValue = {'EW%s' % li:np.zeros(windowSizes.shape) for li in range(1,6)}
fitRMSE = {'EW%s' % li:np.zeros(windowSizes.shape) for li in range(1,6)}
modelCoeffs = {'EW%s' % li:[] for li in range(1,6)}
modelFunctionString = "def modelFunction(x, a, b, c, d, g): return ( (a-d) / ( (1+( (x/c)** b )) **g) ) + d"
exec(modelFunctionString)
noiseVarianceParameters = {'EW%s' % li: {} for li in range(1,6)}
for iSW in range(1,6):
    nnsdHist = npz['nnsdHist'].item()['EW%s' % iSW]
    for li, ws in enumerate(windowSizes):
        snnr = np.array(snnrEdges[:-1] + np.diff(snnrEdges)/2, ndmin=2)
        snnr = np.reshape(np.repeat(snnr, nBins, axis=0), nBins**2)
        nnsd = np.array(nnsdEdges[:-1] + np.diff(nnsdEdges)/2, ndmin=2)
        nnsd = np.reshape(np.repeat(nnsd, nBins, axis=1), nBins**2)
        weight = np.copy(nnsdHist[li]).reshape(nBins**2)
        snnr = snnr[weight!=0]
        nnsd = nnsd[weight!=0]
        weight = weight[weight!=0]
        fitMask = (snnr >= 1.0) * (snnr <= 2.0)
        x = snnr[fitMask]
        y = nnsd[fitMask]
        w = weight[fitMask]/snnr[fitMask]
        pfit = np.polyfit(x, y, w=w, deg=2, full=True)
        fitValue['EW%s' % iSW][li] = np.polyval(pfit[0], 1.0)
        fitRMSE['EW%s' % iSW][li] = np.sqrt(np.sum((w/w.sum())*(y-np.polyval(pfit[0], x))**2))
    modelCoeffs['EW%s' % iSW], pcov = curve_fit(
        modelFunction, windowSizes, fitValue['EW%s' % iSW], sigma=fitRMSE['EW%s' % iSW], maxfev=100000)
    noiseVarianceParameters['EW%s' % iSW] = modelCoeffs['EW%s' % iSW][3]
np.savez_compressed(platform+'_nnsd_results.npz',
         modelFunctionString=modelFunctionString, modelCoeffs=modelCoeffs,
         windowSizes=windowSizes, fitValue=fitValue, fitRMSE=fitRMSE)


npz = np.load(platform+'_extraScaling.npz')
dBsnnrEdges = npz['dBsnnrEdges']
esfEdges = npz['esfEdges']
windowSizes = npz['windowSizes']
nBins = len(dBsnnrEdges)-1
extraScalingParameters = {'EW%s' % li:np.zeros(nBins) for li in range(1,6)}
extraScalingParameters['SNNR'] = dBsnnrEdges[:-1] + np.diff(dBsnnrEdges)/2.
fitValue = {'EW%s' % li:np.zeros((len(windowSizes),nBins)) for li in range(1,6)}
fitRMSE = {'EW%s' % li:np.zeros(windowSizes.shape) for li in range(1,6)}
modelCoeffs = {'EW%s' % li:[] for li in range(1,6)}
def modelFunction(x, a, k, b, v, q, c):
    return a+( (k-a) / (c+q*np.exp(-b*x))**(1/v) )
for iSW in range(1,6):
    esfHist = npz['esfHist'].item()['EW%s' % iSW]
    for li, ws in enumerate(windowSizes):
        dBsnnr = np.array(dBsnnrEdges[:-1] + np.diff(dBsnnrEdges)/2, ndmin=2)
        dBsnnr = np.reshape(np.repeat(dBsnnr, nBins, axis=0), nBins**2)
        esf = np.array(esfEdges[:-1] + np.diff(esfEdges)/2, ndmin=2)
        esf = np.reshape(np.repeat(esf, nBins, axis=1), nBins**2)
        weight = np.copy(esfHist[li]).reshape(nBins**2)
        dBsnnr = dBsnnr[weight!=0]
        esf = esf[weight!=0]
        weight = weight[weight!=0]
        esf[dBsnnr>=3] = 1.0
        weight[dBsnnr<=-1] = 0
        fitMask = (weight >= 1)
        x = dBsnnr[fitMask]
        y = esf[fitMask]
        w = 1/weight[fitMask]
        popt, pcov = curve_fit(modelFunction, x, y, sigma=w, p0=[100,1,1,1,0.005,1], maxfev=10000)
        fittedCurve = modelFunction(extraScalingParameters['SNNR'] , *popt)
        fitValue['EW%s' % iSW][li] = fittedCurve - fittedCurve.min() + 1
        fitRMSE['EW%s' % iSW][li] = np.sqrt(np.sum((w/w.sum())*(y-modelFunction(x, *popt))**2))
    extraScalingParameters['EW%s' % iSW] = np.sum(
        fitValue['EW%s' % iSW].T * (1/fitRMSE['EW%s' % iSW]) / sum(1/fitRMSE['EW%s' % iSW]),axis=1)
np.savez_compressed(platform+'_esf_results.npz', extraScalingParameters=extraScalingParameters)
