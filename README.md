# Sentinel1Denoised
Thermal noise subtraction, scalloping correction, angular correction

# Citation

If you use Sentinel1Denoised in any academic work then you *must* cite the following paper:

Park, Jeong-Won; Korosov, Anton; Babiker, Mohamed; Sandven, Stein; and Won, Joong-Sun (2018): Efficient noise removal of Sentinel-1 TOPSAR cross-polarization channel, IEEE Transactions on Geoscience and Remote Sensing, doi:10.1109/TGRS.2017.2765248

See the CITATION file for more information.

# Installation
The software is written in Python and requires
[nansat](https://nansat.readthedocs.io/en/latest/source/installation.html)
and [scipy](https://www.scipy.org/install.html) packages. A simple way to install these packages
is to use [Anaconda](https://docs.conda.io/en/latest/miniconda.html).

```
# clone repository
git clone https://github.com/nansencenter/sentinel1denoised.git

# create conda environment with key requirements
conda create -y -n py3s1denoise gdal numpy pillow netcdf4 scipy

# activate environment
source activate py3s1denoise

# install nansat
pip instal nansat

# install s1denoise package
python setup.py install
```

Alternatively you can use [Docker](https://www.docker.com/):

```
# build an image with eveything installed
docker build . -t s1denoise

# run Python in container
docker run --rm -it -v /path/to/data:/path/to/data s1denoise python

```

# Example

Do processing inside Python environment:
```python
from s1denoise import Sentinel1Image
# open access to file with S1 data
s1 = Sentinel1Image('/path/to/data/S1B_EW_GRDM_1SDH_INPUTFILE.zip')

# run denoising of HV polarisoation
s1.add_denoised_band('HV')

# get array with denoised values
s0_hv = s1['sigma0_HV_denoised']
```

Process a single file and export as GeoTIFF:

`s1_thermal_denoise.py /path/to/data/S1B_EW_GRDM_1SDH_INPUTFILE.zip /path/to/data/output.tif`

Process a single file, convert to sigma0 to dB, replace negative values with the closest
minimum positive value, process both HH and HV

`s1_thermal_denoise.py INPUTFILE.zip OUTPUTFILE.tif -db -nn -p HHHV`


Process a single file using Docker (replace `input_dir` and `output_dir` with actual directories):

`docker run --rm -it -v /input_dir:/input_dir -v /output_dir:/output_dir s1denoise s1_thermal_denoise.py /input_dir/INPUTFILE.zip /output_dir/OUPUTFILE.tif`
