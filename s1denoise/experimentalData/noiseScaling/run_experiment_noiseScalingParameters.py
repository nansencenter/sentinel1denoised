import os
import sys
import glob
import shutil
from multiprocessing import Pool
from s1denoise import Sentinel1Image
from sys import exit

# Instrument
instr = sys.argv[1]

# Polarization
polarization = sys.argv[2]

if not instr in ['S1A', 'S1B']:
    print('The input data must be S1A or S1B')
    exit()

# Input directory with S1 files
inputPath = sys.argv[3]  #'/mnt/sverdrup-2/sat_auxdata/denoise/dolldrums/zip

# Output directory for storing statistics for individual files
outputPath = sys.argv[4] #'/mnt/sverdrup-2/sat_auxdata/denoise/coefficients_training/noise_scaling'

zipFilesAll = []

zip_ffiles = glob.glob('%s/*%s*.zip' % (inputPath, instr))

for ifile in zip_ffiles:
    zipFilesAll.append(ifile)

try:
    os.makedirs(outputPath)
except:
    pass

def run_process(zipFile):
    outputFilename = zipFile.split('/')[-1].split('.')[0] + '_noiseScaling.npz'

    if os.path.exists(outputPath + outputFilename):
        print('Processed data file already exists.')
    else:
        Sentinel1Image(zipFile).experiment_noiseScaling(polarization)
        shutil.move(outputFilename, outputPath)

zipFilesUnprocessed = [z for z in zipFilesAll
    if not os.path.exists(outputPath + z.split('/')[-1].split('.')[0] + '_noiseScaling.npz')]

print(zipFilesUnprocessed)

pool = Pool(4)
pool.map(run_process, zipFilesUnprocessed)
pool.close(); pool.terminate(); pool.join();
