import os
import sys
import glob
import shutil
from multiprocessing import Pool
from s1denoise import Sentinel1Image

# dir path to noise scaling training data
inputPath = sys.argv[1]
outputPath = sys.argv[2]

'''
with open(inputPath, 'r') as f:
   data = f.readlines()

zipFilesAll = []
'''

instrument = 'S1A'

zipFilesAll = glob.glob('%s/*%s*.zip' % (inputPath, instrument))

# base directory path containing S1 zip files
#base_zip_path = '/mnt/sverdrup-2/sat_auxdata/denoise/norway_nordic_sea/s1/ew/zip/201906261000-202002041121'

#for line in data:
#    zipFilesAll.append('%s/%s' % (base_zip_path, line.replace('\n','')))

# try make directory for output npz files
try:
    os.makedirs(outputPath)
except:
    pass

#inputPath = '/Volumes/MacOS8TB/Archives/Sentinel-1/NorthAtlanticOcean/'
#outputPath = '/Volumes/MacOS8TB/Process/sentinel1denoised/experimentalData/powerBalancing/'

def run_process(zipFile):
    outputFilename = zipFile.split('/')[-1].split('.')[0] + '_powerBalancing.npz'
    print(outputPath + outputFilename)
    if os.path.exists(outputPath + outputFilename):
        print('Processed data file already exists.')
    else:
        Sentinel1Image(zipFile).experiment_powerBalancing('HV')
        print('Done! Moving file to\n %s\%s\n' % (outputFilename, outputPath))
        print('\n#### Moving file...\n')
        shutil.move(outputFilename, outputPath)

#zipFilesAll = sorted(glob.glob('%s/S1?_EW_GRDM_1SDH_*.zip' % (inputPath)), reverse=True)

zipFilesUnprocessed = [z for z in zipFilesAll
    if not os.path.exists(outputPath + os.path.basename(z).split('.')[0] + '_powerBalancing.npz')]

pool = Pool(2)
pool.map(run_process, zipFilesUnprocessed)
pool.close(); pool.terminate(); pool.join();
