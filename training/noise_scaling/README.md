The directory contains scripts for the noise scaling training. It is highly reccomended that the
training should be performed over tens of Sentinel-1 Level 1 GRD files. Due to the antenna gain
differences of S-1A and S-1B, the training should be performed for the platforms separately.

## Step 1. Individual file processing

See ../run_experiment.py


## Step 2. Statistical aggregation

Once you have statistics for many files you can obtain statistically aggregated noise scaling
factors for each sub-swath by scrpit called 'analyze_experiment_noiseScalingParameters.py':

```
python analyze_experiment_noiseScalingParameters.py S1A IW GRDH 1SDV /path/to/npz /path/to/output/file
```

where the arguments:\
1st - platform (S1A/S1B)\
2nd - mode (EW/IW)\
3d  - polarization mode (1SDH/1SDV)\
4th - path to input training npz files from the first step\
5th - path to output file

All steps are described in a Python's notebook in line by line manner:

```python
Quick_start_Noise_scaling.ipynb
```
