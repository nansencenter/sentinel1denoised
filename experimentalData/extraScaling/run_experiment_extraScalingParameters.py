import os, sys, glob, shutil
from multiprocessing import Pool
from sentinel1denoised.S1_TOPS_GRD_NoiseCorrection import Sentinel1Image

inputPath = '/Volumes/MacOS8TB/Archives/Sentinel-1/NorthAtlanticOcean/'
outputPath = '/Volumes/MacOS8TB/Process/sentinel1denoised/experimentalData/extraScaling/'

def run_process(zipFile):
    outputFilename = zipFile.split('/')[-1].split('.')[0] + '_extraScaling.npz'
    print(outputFilename)
    if os.path.exists(outputPath + outputFilename):
        print('Processed data file already exists.')
    else:
        Sentinel1Image(zipFile).experiment_extraScaling('HV', windowSize=25)
        shutil.move(outputFilename, outputPath)

zipFilesAll = sorted(glob.glob(inputPath + 'S1?_EW_GRDM_1SDH_*.zip'),reverse=True)
zipFilesUnprocessed = [z for z in zipFilesAll
    if not os.path.exists(outputPath + z.split('/')[-1].split('.')[0] + '_extraScaling.npz')]
pool = Pool(2)
pool.map(run_process, zipFilesUnprocessed)
