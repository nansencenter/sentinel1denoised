# Sentinel1Denoised
Thermal noise subtraction, scalloping correction, angular correction

# Citation

If you use Sentinel1Denoised in any academic work then you *must* cite the following paper:

Park, Jeong-Won; Korosov, Anton; Babiker, Mohamed; Sandven, Stein; and Won, Joong-Sun (2018): Efficient noise removal of Sentinel-1 TOPSAR cross-polarization channel, IEEE Transactions on Geoscience and Remote Sensing, doi:10.1109/TGRS.2017.2765248

See the CITATION file for more information.

# Installation
The software is written in Python and requires (nansat)[https://nansat.readthedocs.io/en/latest/source/installation.html]
and (scipy)[https://www.scipy.org/install.html] packages. After installing these packages run:
`python setup.py install`

Alternatively you can use (Docker)[https://www.docker.com/] to build an image with eveything installed:
`docker build . -t s1denoise`

# Example
To start Python in Docker run `docker run --rm -it -v /path/to/data:/path/to/data s1denoise python`

To do processing in Python run:
```python
from s1denoise import Sentinel1Image
# open access to file with S1 data
s1 = Sentinel1Image('/path/to/data/S1B_EW_GRDM_1SDH_INPUTFILE.zip')
# run denoising of HV polarisoation
s1.add_denoised_band('HV')
# get array with denoised values
s0_hv = s1['sigma0_HV_denoised']
```

To process a single file and export as GeoTIFF:
`docker run --rm -v /path/to/data:/path/to/data s1denoise s1_thermal_denoise.py /path/to/data/S1B_EW_GRDM_1SDH_INPUTFILE.zip /path/to/data/output`
