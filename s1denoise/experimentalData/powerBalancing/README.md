The directory contains scripts for the power balancing training. It is highly reccomended that the training should be performed over tens of Sentinel-1 Level 1 GRD files. Due to the antenna gain differences of S-1A and S-1B, the training should be performed for the platforms separetely.

Firs, you shoud loop over the trainig files by
```python
run_experiment_powerBalancingParameters.py
```
script. 

Once you have agregated statistics for many files you can obtain robust the power balancing factors by 

```python
analyze_experiment_powerBalancingParameters.py
```

All steps are described in Python's notebook in almost line by line manner:

```python
Quick_start_Power_balancing.ipynb
```
