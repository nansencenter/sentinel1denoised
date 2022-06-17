# Sentinel1Denoised
Thermal noise subtraction, scalloping correction, angular correction

## Citation

If you use Sentinel1Denoised in any academic work then you *must* cite the following paper:

Park, Jeong-Won; Korosov, Anton; Babiker, Mohamed; Sandven, Stein; and Won, Joong-Sun (2018): Efficient noise removal of Sentinel-1 TOPSAR cross-polarization channel, IEEE Transactions on Geoscience and Remote Sensing, 56(3), 1555-1565, doi:10.1109/TGRS.2017.2765248

Park, Jeong-Won; Won, Joong-Sun; Korosov, Anton A.; Babiker, Mohamed; and Miranda, Nuno (2019), Textural Noise Correction for Sentinel-1 TOPSAR Cross-Polarization Channel Images, IEEE Transactions on Geoscience and Remote Sensing, 57(6), 4040-4049, doi:10.1109/TGRS.2018.2889381


See the CITATION file for more information.

## Installation
The software is written in Python and requires
[nansat](https://nansat.readthedocs.io/en/latest/source/installation.html)
and [scipy](https://www.scipy.org/install.html) packages. A simple way to install these packages
is to use [Anaconda](https://docs.conda.io/en/latest/miniconda.html).

```
# create conda environment with key requirements
conda create -y -n s1denoise gdal cartopy pip

# activate environment
conda activate s1denoise

# install other reqs using pip
pip install pythesint netcdf4 nansat

# update metadata vocabularies
python -c 'import pythesint as pti; pti.update_all_vocabularies()'

# install s1denoise
pip install https://github.com/nansencenter/sentinel1denoised/archive/v1.3.1.tar.gz

```

Alternatively you can use [Docker](https://www.docker.com/):

```
# build an image with eveything installed
docker build . -t s1denoise

# run Python in container
docker run --rm -it -v /path/to/data:/path/to/data s1denoise python

```

## Example

Do processing inside Python environment:
```python
from s1denoise import Sentinel1Image
# open access to file with S1 data
s1 = Sentinel1Image('/path/to/data/S1B_EW_GRDM_1SDH_INPUTFILE.zip')

# run thermal noise correction in HV polarisation with the default ESA algorithm
s0hve = s1.remove_thermal_noise('HV', algorithm='ESA')

# run thermal noise correction in HV polarisation with the NEW algorithm
s0_hv = s1.remove_thermal_noise('HV')

# run thermal and texture noise correction in HV polarisation
s0_hv = s1.remove_texture_noise('HV')


```

Process a single file with thermal, textural and angular correction and export in dB

`s1_correction.py INPUTFILE.zip OUTPUTFILE.tif`

Process a single file using Docker (replace `input_dir` and `output_dir` with actual directories):

`docker run --rm -v /input_dir:/input_dir -v /output_dir:/output_dir s1denoise s1_correction.py /input_dir/INPUTFILE.zip /output_dir/OUPUTFILE.tif`

## Experimental scripts

Sub-directories in `s1denoise/experimentalData` contain scripts for training the noise scaling and power balancing coefficients and extra scaling.
See README files in these sub-dirs for details.

## License
The project is licensed under the GNU general public license version 3.
