""" This python script process individual S1 Level-1 GRD files
to get statistics for each sub-block

run example:
python run_experiment_powerBalancingParameters.py S1A VH /path/to/L1/GRD/files /path/to/output/dir
"""

import os
import sys
import glob
import shutil
from multiprocessing import Pool
from s1denoise import Sentinel1Image
from sys import exit

# Instrument
instrument = sys.argv[1]

# Polarization
polarization = sys.argv[2]

if not instrument in ['S1A', 'S1B']:
    print('The input data must be S1A or S1B')
    exit()

# Input directory with S1 L1 GRD files
inputPath = sys.argv[3]

# Output directory for storing statistics for individual files
outputPath = sys.argv[4]

zipFilesAll = glob.glob('%s/*%s*.zip' % (inputPath, instrument))

# try make directory for output npz files
try:
    os.makedirs(outputPath)
except:
    pass

def run_process(zipFile):
    outputFilename = zipFile.split('/')[-1].split('.')[0] + '_powerBalancing.npz'
    print(outputPath + outputFilename)
    if os.path.exists(outputPath + outputFilename):
        print('Processed data file already exists.')
    else:
        Sentinel1Image(zipFile).experiment_powerBalancing(polarization)
        print('Done! Moving file to\n %s\%s\n' % (outputFilename, outputPath))
        print('\n#### Moving file...\n')
        shutil.move(outputFilename, outputPath)

zipFilesUnprocessed = [z for z in zipFilesAll
    if not os.path.exists(outputPath + os.path.basename(z).split('.')[0] + '_powerBalancing.npz')]

pool = Pool(2)
pool.map(run_process, zipFilesUnprocessed)
pool.close(); pool.terminate(); pool.join();