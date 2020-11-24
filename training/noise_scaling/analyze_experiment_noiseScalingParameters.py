#!/usr/bin/env python
""" Process aggregated statistics from individual npz files to obtain final results in noise scaling stage

run example:
python analyze_experiment_noiseScalingParameters.py S1A IW GRDH 1SDV /path/to/npz/files /out/path
"""
import datetime
import glob
import os
import sys

import numpy as np

from s1denoise.utils import AnalyzeExperiment


class AnalyzeNoseScaling(AnalyzeExperiment):
    file_suffix = 'noiseScaling'

    def process(self):
        # stack processed files
        IPFversion = {'%s' % li: [] for li in self.swath_names}
        powerDifference = {'%s' % li: [] for li in self.swath_names}
        scalingFactor = {'%s' % li: [] for li in self.swath_names}
        correlationCoefficient = {'%s' % li: [] for li in self.swath_names}
        fitResidual = {'%s' % li: [] for li in self.swath_names}
        acqDate = {'%s' % li: [] for li in self.swath_names}
        npz_files_per_block = []
        for npzFile in self.npzFiles:
            print('importing %s' % npzFile)
            npz = np.load(npzFile)
            npz.allow_pickle=True

            for iSW in self.swath_names:
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
            for li in range(numberOfSubblocks):
                npz_files_per_block.append(os.path.basename(npzFile))

        # compute fit values
        noiseScalingParameters = {'%s' % li: {} for li in self.swath_names}
        noiseScalingParametersRMSE = {'%s' % li: {} for li in self.swath_names}
        ns_params = {}
        ns_key = f'{self.args.platform}_{self.args.mode}_{self.args.res}_{self.polarisation}_NS'
        for IPFv in np.arange(2.4, 4.0, 0.1):
            ipf_str = '%.1f' % IPFv
            ipf_key = f'{ns_key}_{ipf_str}'
            for iSW in self.swath_names:
                if IPFv==2.7 and self.args.platform=='S1B':
                    valid = np.logical_and(np.array(IPFversion['%s' % iSW])==2.72,
                                           np.array(acqDate['%s' % iSW]) < datetime.datetime(2017,1,16,13,42,34) )
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
                if ipf_key not in ns_params:
                    ns_params[ipf_key] = {'mean': {}, 'rmse': {}}
                ns_params[ipf_key]['mean'][iSW] = fitResults[0]
                ns_params[ipf_key]['rmse'][iSW] = np.sqrt(np.sum((fitResults[0]-sf)**2 * w) / np.sum(w))
                valid_npz_files = list(set([npz_files_per_block[i] for i in np.where(valid)[0]]))
                ns_params[ipf_key]['files'] = self.npzFiles

                noiseScalingParameters[iSW]['%.1f' % IPFv] = fitResults[0]
                noiseScalingParametersRMSE[iSW]['%.1f' % IPFv] = np.sqrt(np.sum((fitResults[0]-sf)**2 * w) / np.sum(w))

        self.save(ns_params)

if __name__ == "__main__":
    ans = AnalyzeNoseScaling()
    ans.process()
