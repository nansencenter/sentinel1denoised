The directory contains scripts for the power balancing training. It is highly reccomended that the training should be performed over tens of Sentinel-1 Level 1 GRD files. Due to the antenna gain differences of S-1A and S-1B, the training should be performed for the platforms separetely.

Firs, you shoud loop over the trainig files by this script called run_experiment_powerBalancingParameters.py. Here is an example how to run it in Ipython shell:

```python
run analyze_experiment_powerBalancingParameters.py S1A /path/to/training/files /out/path
```

where arguments:
1st - platform (S1A/S1B)
2nd - path to training Level-1 GRD data
3d  - path to output files with individual statistics

Once you have agregated statistics for many files you can obtain final power balancing factors by scrpit analyze_experiment_powerBalancingParameters.py:

```python
run analyze_experiment_powerBalancingParameters.py S1A IW GRDH 1SDV /path/to/training/files /out/path
```

All steps are described in a Python's notebook in line by line manner:

```python
Quick_start_Power_balancing.ipynb
```
