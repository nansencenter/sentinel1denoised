#!/usr/bin/env python

""" This python script process individual S1 Level-1 GRD files to get values of antenna pattern gain (APG), sigma0, incidence angle for each noise vector

run example:
python collect_apg_data.py /path/to/L1/GRD/files /path/to/output/dir

"""
import argparse
from collections import defaultdict
import os
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np

from s1denoise import Sentinel1Image

def read_kproc(s1):
    soup = BeautifulSoup(s1.annotationXML['HV'].toxml())
    kprop = {spp.swath.text:  float(spp.processorscalingfactor.text)
             for spp in soup.find_all('swathprocparams')}    
    return kprop

def get_pgpp_pgpa(s1):
    soup = BeautifulSoup(s1.annotationXML['HV'].toxml())
    pgpp = soup.find_all('pgproductphase')
    pgpa = soup.find_all('pgproductamplitude')
    pha_dict = defaultdict(list)
    amp_dict = defaultdict(list)

    for pg in pgpp:
        pha_dict[pg.parent.parent.parent.swath.text].append(float(pg.text))
    for pg in pgpa:
        amp_dict[pg.parent.parent.parent.swath.text].append(float(pg.text))
    return pha_dict, amp_dict

def get_mean_pgpp_pgpa(s1):
    pha_dict, amp_dict = get_pgpp_pgpa(s1)
    pha_dict_avg = {i: np.mean(pha_dict[i]) for i in pha_dict}
    amp_dict_avg = {i: np.mean(amp_dict[i]) for i in amp_dict}
    return pha_dict_avg, amp_dict_avg

def parse_run_experiment_args():
    """ Parse input args for run_experiment_* scripts """
    parser = argparse.ArgumentParser(description='Process SAFE or ZIP files and collect APG related values')
    parser.add_argument('-o', '--out-dir', type=Path)
    parser.add_argument('-f', '--force', action='store_true', help="Force overwrite existing output files")
    parser.add_argument('-i', '--inp-files', type=Path, nargs='+')

    return parser.parse_args()

def main():
    polarization = 'HV'
    args = parse_run_experiment_args()
    print('Number of files queued:', len(args.inp_files))

    for ifile in args.inp_files:
        ofile = os.path.join(args.out_dir, os.path.basename(ifile) + '_apg.npz')
        if os.path.exists(ofile) and not args.force:
            print(ofile, 'exists')
            continue

        try:
            s1 = Sentinel1Image(str(ifile))
            dn_hv = s1.get_GDALRasterBand('DN_HV').ReadAsArray().astype(float)
        except RuntimeError:
            print('GDAL cannot open ', ifile)
            continue
        print('Process ', ifile)

        line, pixel, noise = s1.get_noise_range_vectors(polarization)
        swath_ids = s1.get_swath_id_vectors(polarization, line, pixel)
        eap, rsl = s1.get_eap_rsl_vectors(polarization, line, pixel)
        ea_interpolator, ia_interpolator = s1.get_elevation_incidence_angle_interpolators(polarization)
        eleang, incang = s1.get_elevation_incidence_angle_vectors(ea_interpolator, ia_interpolator, line, pixel)
        cal_s0hv = s1.get_calibration_vectors(polarization, line, pixel)
        scall_hv = s1.get_noise_azimuth_vectors(polarization, line, pixel)

        dn_hv[dn_hv < 20] = np.nan
        sigma0hv = s1.get_raw_sigma0_vectors_from_full_size(line, pixel, swath_ids, dn_hv)

        pgpp, pgpa = get_pgpp_pgpa(s1)
        kproc = read_kproc(s1)
        eap_xml = s1.import_elevationAntennaPattern('HV')
        ncf = {i: eap_xml[i]['noiseCalibrationFactor'] for i in eap_xml}
        acc = eap_xml['EW1']['absoluteCalibrationConstant']
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
            sigma0hv=sigma0hv,
            pgpp=pgpp,
            pgpa=pgpa,
            kproc=kproc,
            ncf=ncf,
            acc=acc,
            ipf=ipf,
        )

if __name__ == "__main__":
    main()
