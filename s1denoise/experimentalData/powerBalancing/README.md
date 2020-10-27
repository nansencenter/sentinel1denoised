The directory contains scripts for the power balancing training. It is highly reccomended that the training should be performed over tens of Sentinel-1 Level 1 GRD files. Due to the antenna gain differences of S-1A and S-1B, the training should be performed for the platforms separetely.

##1st step. Individual training files processing 

The script run_experiment_powerBalancingParameters.py loop over the trainig S1 Level 1 GRD files. Here is an example how to run it via IPython shell:

```python
run analyze_experiment_powerBalancingParameters.py S1A /path/to/training/files /out/path/npz
```
where arguments:\
1st - platform (S1A/S1B)\
2nd - path to input training Level-1 GRD data\
3d  - path to output files with individual statistics

##2nd stage. Aggregated statistics processing

Once you have agregated statistics for many files you can obtain final power balancing factors by scrpit analyze_experiment_powerBalancingParameters.py:

```python
run analyze_experiment_powerBalancingParameters.py S1A IW GRDH 1SDV /path/to/npz/files /out/path
```
where arguments:\
1st - platform (S1A/S1B)\
2nd - mode (EW/IW)\
3d  - polarization mode (1SDH/1SDV)\
4th - path to input training npz files from the first step\
5th - path to output dir

All steps are described in a Python's notebook in line by line manner:

```python
Quick_start_Power_balancing.ipynb
```
