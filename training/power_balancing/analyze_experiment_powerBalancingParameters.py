#!/usr/bin/env python
""" Process aggregated statistics from individual npz files to obtain final results in power balancing stage

run example:
python analyze_experiment_powerBalancingParameters.py S1A IW GRDH 1SDV /path/to/npz/files /out/path
"""

import json
import datetime
import glob
import os
import sys

import numpy as np

from s1denoise.utils import AnalyzeExperiment


class AnalyzePowerBalancing(AnalyzeExperiment):
    file_suffix = 'powerBalancing'

    def process(self):
        # stack data from pre-processed files
        IPFversion = []
        powerDifference = []
        balancingPower = []
        correlationCoefficient = []
        fitResidual = []
        acqDate = []
        npz_files_per_block = []
        for npzFile in self.npzFiles:
            print('importing %s' % npzFile)
            npz = np.load(npzFile)
            npz.allow_pickle = True

            numberOfSubblocks = np.unique([ len(npz[iSW].item()['balancingPower'])
                                            for iSW in self.swath_names])
            if numberOfSubblocks.size != 1:
                print('*** numberOfSubblocks are not consistent for all subswaths.')
                continue
            numberOfSubblocks = numberOfSubblocks.item()

            for li in range(numberOfSubblocks):
                powerDifference.append([
                      np.nanmean(10*np.log10(npz[iSW].item()['sigma0'][li]))
                    - np.nanmean(10*np.log10(npz[iSW].item()['noiseEquivalentSigma0'][li]))
                    for iSW in self.swath_names])
                balancingPower.append([
                    npz[iSW].item()['balancingPower'][li]
                    for iSW in self.swath_names])
                correlationCoefficient.append([
                    npz[iSW].item()['correlationCoefficient'][li]
                    for iSW in self.swath_names])
                fitResidual.append([
                    npz[iSW].item()['fitResidual'][li]
                    for iSW in self.swath_names])
                IPFversion.append(npz['IPFversion'])
                acqDate.append(datetime.datetime.strptime(os.path.basename(npzFile).split('_')[4], '%Y%m%dT%H%M%S'))
                npz_files_per_block.append(os.path.basename(npzFile))

        powerDifference = np.array(powerDifference)
        balancingPower = np.array(balancingPower)
        correlationCoefficient = np.array(correlationCoefficient)
        fitResidual = np.array(fitResidual)
        IPFversion = np.array(IPFversion)
        acqDate = np.array(acqDate)

        # compute fit values
        pb_params = {}
        pb_key = f'{self.args.platform}_{self.args.mode}_{self.args.res}_{self.polarisation}_PB'
        for IPFv in np.arange(2.4, 4.0, 0.1):
            if IPFv==2.7 and self.args.platform=='S1B':
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
            ipf_str = '%.1f' % IPFv
            ipf_key = f'{pb_key}_{ipf_str}'
            pb_params[ipf_key] = {'mean': {}, 'rmse': {}}
            for iSW in range(0, self.swaths_number):
                bp = balancingPower[valid][:,iSW]
                fitResults = np.polyfit(pd, bp, deg=0, w=w)
                pb_params[ipf_key]['mean'][self.swath_names[iSW]] = fitResults[0]
                pb_params[ipf_key]['rmse'][self.swath_names[iSW]] = np.sqrt(np.sum((fitResults[0]-bp)**2 * w) / np.sum(w))
            valid_npz_files = list(set([npz_files_per_block[i] for i in np.where(valid)[0]]))
            pb_params[ipf_key]['files'] = valid_npz_files

        self.save(pb_params)

if __name__ == "__main__":
    apb = AnalyzePowerBalancing()
    apb.process()
