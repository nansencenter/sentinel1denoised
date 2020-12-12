# Quality assessment of thermal noise removal in range and azimuth directions for multi-swath S1 images 

This directory contain scripts for quality assessment of thermal noise removal from S1-A/B Level-1 GRD images. The quality assessment comprises the proposed quality metric calculation for range and azimuth directions. The range quality metric (RQM) is based on Fisher's criteria and allows to estimate how a noise removal algorithm reduce discontinuities in sub-swaths for multi-swath acquisition - the less value of RQM means more successful noise removal.
The azimuth quality metric (AQM) is based on periodicity detection in the signal using the auto-correlation function and a simple indication of the existence of a prominent scalloping effect.

To perform the validation you need:

1. Run the script ``` python run_qm.py rqm /path/to/L1/GRD/files /path/to/output/dir ``` to caclulate RQM for individual files
2. Run the script ``` python rqm_plot.py input/npz/path output/path ``` to plot the averaged statistics for each region as an aggregated bar plot
3. Run the script ``` python rqm_tables.py input/npz/path output/path ``` to generate latex-tables of the obtained statistics
