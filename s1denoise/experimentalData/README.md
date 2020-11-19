The directory contains generic scripts for working with coefficiens.

## 1. Convert from old format (needed only once):

`python convert_old_parameters.py`

It converts from two JSON files for S1A and S1B in old format into one file in new format.
In the new format the dict structure is quite flat. There are only two levels. The root key defines
platform, mode, resolution, polarisation, type of coefficient and IPF (e.g. `S1B_EW_GRDM_HV_NS_3.1`).
The 2nd next level key defines sub-swath name (e.g. `EW1`).

Coefficient can be access by:
```python
import json
with open('denoising_parameters.json') as f:
    par = json.load(f)

ew1_ns = par['S1B_EW_GRDM_HV_NS_3.1']['EW1']
ew1_pb = par['S1B_EW_GRDM_HV_PB_3.1']['EW1']
```


## 2. Processing individual S1 GRD files with noise scaling and power balancing experiments

The script 'run_experiment.py' loops over a set of S1 Level 1 GRD files to get statistics for
each sub-block. Here is an example how to run it:

```
python run_experiment.py ns S1A VH /path/to/L1/GRD/files /path/to/output/dir
```
where the arguments:\
1st - type of experiment (ns/pb)
2st - platform (S1A/S1B)\
3nd - polarization (VH/HV)\
4nd - path to input training Level-1 GRD data\
5th - path to output npz files with statistics in sub-blocks


## 3. Aggreagate statistics from individual NPZ files

See noiseScaling and powerBalancing directories

## 4. Update the main denoising_parameters.json file with values from an experiment:

Once you have statistics for many files you can obtain statistically aggregated power balancing factors for each sub-swath by scrpit called 'analyze_experiment_powerBalancingParameters.py':

```
python update_parameter_file.py powerBalancing/S1B_EW_GRDM_1SDH_power_balancning.json ../denoising_parameters.json
```

with the arguments:
1. name of input file with averaged values of power balancing coefficients (output from `analyze_experiment_powerBalancingParameters.py`)
1. name of the main parameters file

It will take the averaged values from input file and put them into the main file. In addition it will
create file `denoising_parameters_training_files.json` with lists of all training files sued in that experiment.

Both `denoising_parameters.json` and `denoising_parameters_training_files.json` should be added to git.
