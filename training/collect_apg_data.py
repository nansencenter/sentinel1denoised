#!/usr/bin/env python

""" This python script process individual S1 Level-1 GRD files to get values of antenna pattern gain (APG), sigma0, incidence angle for each noise vector

run example:
python collect_apg_data.py /path/to/L1/GRD/files /path/to/output/dir

"""
import argparse
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from s1denoise import Sentinel1Image

def get_processor_scaling_factor(soup):
    k_proc = {spp.swath.text:  float(spp.processorscalingfactor.text)
             for spp in soup.find_all('swathprocparams')}    
    return k_proc

def get_relative_azimuth_time(s1, polarization, line):
    geolocationGridPoint = s1.import_geolocationGridPoint(polarization)
    xggp = np.unique(geolocationGridPoint['pixel'])
    yggp = np.unique(geolocationGridPoint['line'])

    azimuth_time = [ (t-s1.time_coverage_center).total_seconds() for t in geolocationGridPoint['azimuthTime'] ]
    azimuth_time = np.reshape(azimuth_time, (len(yggp), len(xggp)))
    at_interp = InterpolatedUnivariateSpline(yggp, azimuth_time[:,0])
    azimuth_time = at_interp(line)
    return azimuth_time

def get_pg_product(s1, soup, azimuth_time, pg_name='pgproductamplitude'):
    pg = defaultdict(dict)
    for pgpa in soup.find_all(pg_name):
        pg[pgpa.parent.parent.parent.swath.text][pgpa.parent.azimuthtime.text] = float(pgpa.text)

    pg_swaths = {}
    for swid in pg:
        rel_az_time = np.array([
        (datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') - s1.time_coverage_center).total_seconds()
        for t in pg[swid]])
        pgvec = np.array([pg[swid][i] for i in pg[swid]])
        sortIndex = np.argsort(rel_az_time)
        pg_interp = InterpolatedUnivariateSpline(rel_az_time[sortIndex], pgvec[sortIndex], k=1)
        pg_vec = pg_interp(azimuth_time)
        pg_swaths[swid] = pg_vec
    return pg_swaths

def get_noise_power_correction_factor(s1, soup, azimuth_time):
    npcf = defaultdict(dict)
    for i in soup.find_all('noisepowercorrectionfactor'):
        npcf[i.parent.swath.text][i.parent.azimuthtime.text] = float(i.text)

    npcf_swaths = {}
    for swid in npcf:
        rel_az_time = np.array([
        (datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') - s1.time_coverage_center).total_seconds()
        for t in npcf[swid]])
        npcfvec = np.array([npcf[swid][i] for i in npcf[swid]])
        if npcfvec.size == 1:
            npcf_vec = np.ones(azimuth_time.size) * npcfvec[0]
        else:
            sortIndex = np.argsort(rel_az_time)
            npcf_interp = InterpolatedUnivariateSpline(rel_az_time[sortIndex], npcfvec[sortIndex], k=1)
            npcf_vec = npcf_interp(azimuth_time)
        npcf_swaths[swid] = npcf_vec
    return npcf_swaths

def get_noise_calibration_factor(s1):
    eap_xml = s1.import_elevationAntennaPattern('HV')
    noise_ca_fa = {i: eap_xml[i]['noiseCalibrationFactor'] for i in eap_xml}
    return noise_ca_fa

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Process SAFE or ZIP files and collect APG related values')
    parser.add_argument('-o', '--out-dir', type=Path)
    parser.add_argument('-w', '--overwrite', action='store_true', help="Overwrite existing output files")
    parser.add_argument('-i', '--inp-files', type=Path, nargs='+')

    return parser.parse_args()

def main():
    polarization = 'HV'
    args = parse_run_experiment_args()
    print('Number of files queued:', len(args.inp_files))

    for ifile in args.inp_files:
        ofile = os.path.join(args.out_dir, os.path.basename(ifile) + '_apg.npz')
        if os.path.exists(ofile) and not args.overwrite:
            print(ofile, 'exists')
            continue

        try:
            s1 = Sentinel1Image(str(ifile))
            dn_hv = s1.get_GDALRasterBand('DN_HV').ReadAsArray().astype(float)
        except RuntimeError:
            print('GDAL cannot open ', ifile)
            continue
        print('Process ', ifile)
        annotation_soup = BeautifulSoup(s1.annotationXML[polarization].toxml())
        k_proc = get_processor_scaling_factor(annotation_soup)
        line, pixel, noise = s1.get_noise_range_vectors(polarization)
        az_time = get_relative_azimuth_time(s1, polarization, line)
        pg_amplitude = get_pg_product(s1, annotation_soup, az_time, pg_name='pgproductamplitude')
        noise_po_co_fa = get_noise_power_correction_factor(s1, annotation_soup, az_time)
        noise_ca_fa = get_noise_calibration_factor(s1)

        swath_ids = s1.get_swath_id_vectors(polarization, line, pixel)
        eap, rsl = s1.get_eap_rsl_vectors(polarization, line, pixel, rsl_power=2.)
        ea_interpolator, ia_interpolator = s1.get_elevation_incidence_angle_interpolators(polarization)
        eleang, incang = s1.get_elevation_incidence_angle_vectors(ea_interpolator, ia_interpolator, line, pixel)
        cal_s0hv = s1.get_calibration_vectors(polarization, line, pixel)
        scall_hv = s1.get_noise_azimuth_vectors(polarization, line, pixel)

        dn_hv[dn_hv < 20] = np.nan
        dn_vectors = s1.get_raw_sigma0_vectors_from_full_size(line, pixel, swath_ids, dn_hv)

        ipf = s1.IPFversion

        np.savez(
            ofile, 
            line=line,
            pixel=pixel,
            noise=noise,
            swath_ids=swath_ids,
            eap=eap,
            rsl=rsl,
            eleang=eleang,
            incang=incang,
            cal_s0hv=cal_s0hv,
            scall_hv=scall_hv,
            dn_vectors=dn_vectors,
            pg_amplitude=pg_amplitude,
            noise_po_co_fa=noise_po_co_fa,
            k_proc=k_proc,
            noise_ca_fa=noise_ca_fa,
            ipf=ipf,
        )

if __name__ == "__main__":
    main()
