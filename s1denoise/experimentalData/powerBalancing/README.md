The directory contains scripts for the power balancing training. It is highly reccomended that the training should be performed over tens of Sentinel-1 Level 1 GRD files. Due to the antenna gain differences of S-1A and S-1B, the training should be performed for the platforms separetely.

##1st step. Individual file processing 

The script 'run_experiment_powerBalancingParameters.py' loop over a set of S1 Level 1 GRD files to get statistics for each sub-block. Here is an example how to run it via IPython shell:

```python
run run_experiment_powerBalancingParameters.py S1A VH /mnt/sverdrup-2/sat_auxdata/denoise/dolldrums/zip /mnt/sverdrup-2/sat_auxdata/denoise/coefficients_training/power_balancing/dolldrums
```
where the arguments:\
1st - platform (S1A/S1B)\
2nd - polarization (VH/HV)
3nd - path to input training Level-1 GRD data\
4d  - path to output npz files with statistics in sub-blocks

##2nd stage. Statistical aggregation

Once you have statistics for many files you can obtain statistically aggregated power balancing factors for each sub-swath by scrpit called 'analyze_experiment_powerBalancingParameters.py':

```python
run analyze_experiment_powerBalancingParameters.py S1A IW GRDH 1SDV /path/to/npz/files /out/path
```

where the arguments:\
1st - platform (S1A/S1B)\
2nd - mode (EW/IW)\
3d  - polarization mode (1SDH/1SDV)\
4th - path to input training npz files from the first step\
5th - path to output dir

All steps are described in a Python's notebook in line by line manner:

```python
Quick_start_Power_balancing.ipynb
```
