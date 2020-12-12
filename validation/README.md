The scripts for quality assessment of thermal noise removal from S1-A/B Level-1 GRD images. The quality assessment is based on quality metrics calculation for range and azimuth directions. The range quality metric (RQM) is based on Fisher's criteria t measure the flatness of the signal near sub-swath margins - the less value of RQM means more successful the noise removal. The azimuth quality metric (AQM) is based on periodicity detection in the signal using the auto-correlation function.

To perform the validation you need:

1. Run the script ``` python run_qm.py rqm /path/to/L1/GRD/files /path/to/output/dir ``` to caclulate RQM for individual files
2. Run the script ``` run rqm_plot.py input/npz/path output/path ``` to plot the averaged statistics for each region as an aggregated bar plot
3. Run the script ``` rqm_tables.py ``` to generate latex-tables of the obtained statistics
