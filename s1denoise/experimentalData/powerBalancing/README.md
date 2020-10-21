Quick start for the power balancing stage

This tutorial describe the training procedure for the balancing steps based on experiment_powerBalancing method of the Sentinel1Image class.

1. Training 
First, you need to get a dataset with statistics on balancing power for sub-blocks within subswaths.
To do that you should process a number of Sentinel-1 Level1 GRD files (we reccomend several tens).

For each file we do the following:

Import S1Image class and open an image we want to process
```python
from s1denoise import Sentinel1Image

s1 = Sentinel1Image('$YOUR_DATA_PATH/zip/S1A_IW_GRDH_1SDV_20200607T075151_20200607T075220_032908_03CFD7_9E14.zip')

```

Initialize number of lines for sub-block to get averages in range direction
```python
polarization='VH', numberOfLinesToAverage=1000
```

Cut a number of border pixels of the image. 100 px for IW data or 25 px for EW

```python
cPx = {'IW':100, 'EW':25}[s1.obsMode] 
```

Call subswathIndexMap method to get a matrix consistent with data matrix with sub-swath numbers from 1 to N, where N is the number of sub-swaths

```python
subswathIndexMap = s1.subswathIndexMap(polarization)
```

![alt text](https://github.com/nansencenter/sentinel1denoised/tree/dd_test/s1denoise/experimentalData/powerBalancing/ss.png "Sub-swath index map fro IW data expample")
