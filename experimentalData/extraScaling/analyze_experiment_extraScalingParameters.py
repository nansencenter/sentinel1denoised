import os, sys, glob, datetime
import numpy as np
from scipy.optimize import curve_fit


platform = 'S1A'
#platform = 'S1B'

bins = 1001
snnr_dB_range = np.array([-5, +5], dtype=np.float)
snnr_range = np.array([0, +10], dtype=np.float)
esf_range = np.array([0, +10], dtype=np.float)
nnsd_range = np.array([0, +10], dtype=np.float)

snnr_dB_range = [ snnr_dB_range[0]-(snnr_dB_range[-1]-snnr_dB_range[0])/(bins-1)/2.,
                  snnr_dB_range[-1]+(snnr_dB_range[-1]-snnr_dB_range[0])/(bins-1)/2. ]
snnr_range = [ snnr_range[0]-(snnr_range[-1]-snnr_range[0])/(bins-1)/2.,
               snnr_range[-1]+(snnr_range[-1]-snnr_range[0])/(bins-1)/2. ]
esf_range = [ esf_range[0]-(esf_range[-1]-esf_range[0])/(bins-1)/2.,
              esf_range[-1]+(esf_range[-1]-esf_range[0])/(bins-1)/2. ]
nnsd_range = [ nnsd_range[0]-(nnsd_range[-1]-nnsd_range[0])/(bins-1)/2.,
               nnsd_range[-1]+(nnsd_range[-1]-nnsd_range[0])/(bins-1)/2. ]


# import data
npzFilesAll = sorted(glob.glob('%s_EW_GRDM_1SDH_*_extraScaling.npz' % platform))
# Check quality disclaimer #30 and #31 in https://qc.sentinel1.eo.esa.int/disclaimer/
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
IPF_version = {'EW%s' % li: [] for li in range(1,6)}
esf_hist2d = {'EW%s' % li: np.zeros((len(npzFiles),bins,bins), dtype=np.uint16) for li in range(1,6)}
nnsd_hist2d = {'EW%s' % li: np.zeros((len(npzFiles),bins,bins), dtype=np.uint16) for li in range(1,6)}
acq_date = {'EW%s' % li: [] for li in range(1,6)}
for li, npzFile in enumerate(npzFiles):
    print('importing %s' % npzFile)
    npz = np.load(npzFile)
    for iSW in range(1,6):
        npzSW = npz['EW%s' % iSW].item()
        esf_hist2d['EW%s' % iSW][li], snnr_dB_edges, esf_edges = np.histogram2d(
                                    10*np.log10(npzSW['signalPlusNoiseToNoiseRatio']),
                                    npzSW['extraScalingFactor'],
                                    bins=bins,
                                    range=[snnr_dB_range,esf_range])
        nnsd_hist2d['EW%s' % iSW][li], snnr_edges, nnsd_edges = np.histogram2d(
                                    npzSW['signalPlusNoiseToNoiseRatio'],
                                    npzSW['noiseNormalizedStandardDeviation'],
                                    bins=bins,
                                    range=[snnr_range,nnsd_range])
        IPF_version['EW%s' % iSW].append(npz['IPFversion'])
        acq_date['EW%s' % iSW].append(datetime.datetime.strptime(npzFile.split('_')[4], '%Y%m%dT%H%M%S'))

for iSW in range(1,6):
    np.savez_compressed(platform+'_extraScalingFactor_EW%s.npz' % iSW,
                        IPF_version=IPF_version,
                        acq_date=acq_date,
                        snnr_dB_edges=snnr_dB_edges,
                        esf_edges=esf_edges,
                        esf_hist2d=esf_hist2d['EW%s' % iSW])
    np.savez_compressed(platform+'_noiseNormalizedStandardDeviation_EW%s.npz' % iSW,
                        IPF_version=IPF_version,
                        acq_date=acq_date,
                        snnr_edges=snnr_edges,
                        nnsd_edges=nnsd_edges,
                        nnsd_hist2d=nnsd_hist2d['EW%s' % iSW])



def model_function(x,a,k,b,v,q,c):
    return a+( (k-a) / (c+q*np.exp(-b*x))**(1/v) )
plt.figure(figsize=(14.4,4.8))
extraScalingParameters = {'EW%s' % li: {} for li in range(1,6)}
for iSW in range(1,6):
    npz = np.load(platform + '_extraScalingFactor_EW%s.npz' % iSW)
    snnr_dB_edges = npz['snnr_dB_edges']
    esf_edges = npz['esf_edges']
    esf_hist2d = npz['esf_hist2d']
    extraScalingParameters['SNNR'] = snnr_dB_edges[:-1] + np.diff(snnr_dB_edges)/2.
    snnr_dB = np.array(snnr_dB_edges[:-1] + np.diff(snnr_dB_edges)/2., ndmin=2)
    snnr_dB = np.reshape(np.repeat(snnr_dB, bins, axis=0),bins*bins)
    esf = np.array(esf_edges[:-1] + np.diff(esf_edges)/2., ndmin=2)
    esf = np.reshape(np.repeat(esf, bins, axis=1),bins*bins)
    weight = np.sum(esf_hist2d,axis=0).T
    plt.subplot(1,5,iSW)
    plt.imshow(np.ma.masked_where(weight==0,weight), origin='low', interpolation='none', aspect='auto',
               extent=[snnr_dB_edges[0], snnr_dB_edges[-1], esf_edges[0], esf_edges[-1]] )
    weight = np.reshape(weight, bins*bins)
    snnr_dB = snnr_dB[weight!=0]
    esf = esf[weight!=0]
    weight = weight[weight!=0]
    esf[snnr_dB>=3] = 1.0
    weight[snnr_dB<=-1] = 0
    popt, pcov = curve_fit( model_function, snnr_dB, esf, sigma=1./weight, p0=[100,1,1,1,0.005,1], maxfev=10000 )
    fittedCurve = model_function(extraScalingParameters['SNNR'] , *popt)
    plt.plot(extraScalingParameters['SNNR'], fittedCurve, 'y')
    plt.axis([snnr_dB_edges[0], snnr_dB_edges[-1], esf_edges[0], esf_edges[-1]])
    extraScalingParameters['EW%s' % iSW] = fittedCurve - fittedCurve.min() + 1.
plt.tight_layout()
plt.tight_layout()


plt.figure(figsize=(14.4,4.8))
noiseVarianceParameters = {'EW%s' % li: {} for li in range(1,6)}
for iSW in range(1,6):
    npz = np.load(platform + '_noiseNormalizedStandardDeviation_EW%s.npz' % iSW)
    snnr_edges = npz['snnr_edges']
    nnsd_edges = npz['nnsd_edges']
    nnsd_hist2d = npz['nnsd_hist2d']
    snnr = np.array(snnr_edges[:-1] + np.diff(snnr_edges)/2., ndmin=2)
    snnr = np.reshape(np.repeat(snnr, bins, axis=0),bins*bins)
    nnsd = np.array(nnsd_edges[:-1] + np.diff(nnsd_edges)/2., ndmin=2)
    nnsd = np.reshape(np.repeat(nnsd, bins, axis=1),bins*bins)
    weight = np.sum(nnsd_hist2d,axis=0).T
    plt.subplot(1,5,iSW)
    plt.imshow(np.ma.masked_where(weight==0,weight), origin='low', interpolation='none', aspect='auto',
               extent=[snnr_edges[0], snnr_edges[-1], nnsd_edges[0], nnsd_edges[-1]] )
    weight = np.reshape(weight, bins*bins)
    snnr = snnr[weight!=0]
    nnsd = nnsd[weight!=0]
    weight = weight[weight!=0]
    fitMask = (snnr >= 1.0) * (snnr <= 2.0)
    x = snnr[fitMask]
    y = nnsd[fitMask]
    w = weight[fitMask]/snnr[fitMask]
    pfit1 = np.polyfit(x, y, w=w, deg=1, full=True)
    print(np.polyval(pfit1[0], 1.0),  np.sqrt(np.sum((w/w.sum())*(y-np.polyval(pfit1[0], x))**2))  )
    pfit2 = np.polyfit(x, y, w=w, deg=2, full=True)
    print(np.polyval(pfit2[0], 1.0),  np.sqrt(np.sum((w/w.sum())*(y-np.polyval(pfit2[0], x))**2))  )
    pfit3 = np.polyfit(x, y, w=w, deg=3, full=True)
    print(np.polyval(pfit3[0], 1.0),  np.sqrt(np.sum((w/w.sum())*(y-np.polyval(pfit3[0], x))**2))  )
    noiseVarianceParameters['EW%s' % iSW] = np.polyval(pfit1[0], 1.0)
plt.tight_layout()
plt.tight_layout()
