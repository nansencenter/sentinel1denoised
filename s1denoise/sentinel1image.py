from datetime import datetime, timedelta
import glob
import json
import os
import requests
import shutil
import subprocess
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString
import zipfile

from nansat import Nansat
import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from s1denoise.utils import (
    cost,
    fit_noise_scaling_coeff,
    get_DOM_nodeValue,
    fill_gaps,
)

SPEED_OF_LIGHT = 299792458.


class Sentinel1Image(Nansat):
    """ Cal/Val routines for Sentinel-1 performed on range noise vector coordinatess"""

    def __init__(self, filename, mapper='sentinel1_l1', log_level=30, fast=False):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__(self, filename, mapper=mapper, log_level=log_level, fast=fast)
        if ( self.filename.split(os.sep)[-1][4:16]
             not in [ 'IW_GRDH_1SDH',
                      'IW_GRDH_1SDV',
                      'EW_GRDM_1SDH',
                      'EW_GRDM_1SDV'  ] ):
             raise ValueError( 'Source file must be Sentinel-1A/1B '
                 'IW_GRDH_1SDH, IW_GRDH_1SDV, EW_GRDM_1SDH, or EW_GRDM_1SDV product.' )
        self.platform = self.filename.split(os.sep)[-1][:3]    # S1A or S1B
        self.obsMode = self.filename.split(os.sep)[-1][4:6]    # IW or EW
        pol_mode = os.path.basename(self.filename).split('_')[3]
        self.crosspol = {'1SDH': 'HV', '1SDV': 'VH'}[pol_mode]
        self.pols = {'1SDH': ['HH', 'HV'], '1SDV': ['VH', 'VV']}[pol_mode]
        self.swath_ids = range(1, {'IW':3, 'EW':5}[self.obsMode]+1)
        txPol = self.filename.split(os.sep)[-1][15]    # H or V
        self.annotationXML = {}
        self.calibrationXML = {}
        self.noiseXML = {}

        if zipfile.is_zipfile(self.filename):
            with zipfile.PyZipFile(self.filename) as zf:
                annotationFiles = [fn for fn in zf.namelist() if 'annotation/s1' in fn]
                calibrationFiles = [fn for fn in zf.namelist()
                                    if 'annotation/calibration/calibration-s1' in fn]
                noiseFiles = [fn for fn in zf.namelist() if 'annotation/calibration/noise-s1' in fn]

                for polarization in [txPol + 'H', txPol + 'V']:
                    self.annotationXML[polarization] = parseString(
                            [zf.read(fn) for fn in annotationFiles if polarization.lower() in fn][0])
                    self.calibrationXML[polarization] = parseString(
                        [zf.read(fn) for fn in calibrationFiles if polarization.lower() in fn][0])
                    self.noiseXML[polarization] = parseString(
                        [zf.read(fn) for fn in noiseFiles if polarization.lower() in fn][0])
                self.manifestXML = parseString(zf.read([fn for fn in zf.namelist()
                                                        if 'manifest.safe' in fn][0]))
        else:
            annotationFiles = [fn for fn in glob.glob(self.filename+'/annotation/*') if 's1' in fn]
            calibrationFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*')
                                if 'calibration-s1' in fn]
            noiseFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*')
                          if 'noise-s1' in fn]

            for polarization in [txPol + 'H', txPol + 'V']:

                for fn in annotationFiles:
                    if polarization.lower() in fn:
                        with open(fn) as ff:
                            self.annotationXML[polarization] = parseString(ff.read())

                for fn in calibrationFiles:
                    if polarization.lower() in fn:
                        with open(fn) as ff:
                            self.calibrationXML[polarization] = parseString(ff.read())

                for fn in noiseFiles:
                    if polarization.lower() in fn:
                        with open(fn) as ff:
                            self.noiseXML[polarization] = parseString(ff.read())

            with open(glob.glob(self.filename+'/manifest.safe')[0]) as ff:
                self.manifestXML = parseString(ff.read())

        # scene center time will be used as the reference for relative azimuth time in seconds
        self.time_coverage_center = ( self.time_coverage_start + timedelta(
            seconds=(self.time_coverage_end - self.time_coverage_start).total_seconds()/2) )
        # get processor version of Sentinel-1 IPF (Instrument Processing Facility)
        self.IPFversion = float(self.manifestXML.getElementsByTagName('safe:software')[0]
                                .attributes['version'].value)
        if self.IPFversion < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Denoising vectors in annotation files are not qualified. '
                  'Only APG-based denoising can be performed\n')
        elif 2.43 <= self.IPFversion < 2.53:
            print('\nWARNING: IPF version of input image is lower than 2.53! '
                  'ESA default noise correction result might be wrong.\n')
        # get the auxiliary calibration file
        resourceList = self.manifestXML.getElementsByTagName('resource')
        resourceList += self.manifestXML.getElementsByTagName('safe:resource')
        for resource in resourceList:
            if resource.attributes['role'].value=='AUX_CAL':
                auxCalibFilename = resource.attributes['name'].value.split('/')[-1]
        self.set_aux_data_dir()
        self.download_aux_calibration(auxCalibFilename, self.platform.lower())
        self.auxiliaryCalibrationXML = parse(self.auxiliaryCalibration_file)

    def set_aux_data_dir(self):
        """ Set directory where aux calibration data is stored """
        self.aux_data_dir = os.path.join(os.environ.get('XDG_DATA_HOME', os.path.expanduser('~')),
                                         '.s1denoise')
        if not os.path.exists(self.aux_data_dir):
            os.makedirs(self.aux_data_dir)

    def download_aux_calibration(self, filename, platform):
        self.auxiliaryCalibration_file = os.path.join(self.aux_data_dir, filename, 'data', '%s-aux-cal.xml' % platform)
        if os.path.exists(self.auxiliaryCalibration_file):
            return
        vs = filename.split('_')[3].lstrip('V')
        validity_start = f'{vs[:4]}-{vs[4:6]}-{vs[6:8]}T{vs[9:11]}:{vs[11:13]}:{vs[13:15]}'

        cd = filename.split('_')[4].lstrip('G')
        creation_date = f'{cd[:4]}-{cd[4:6]}-{cd[6:8]}T{cd[9:11]}:{cd[11:13]}:{cd[13:15]}'
        def get_remote_url(api_url):
            with requests.get(api_url, stream=True) as r:
                rjson = json.loads(r.content.decode())
                remote_url = rjson['results'][0]['remote_url']
                physical_name = rjson['results'][0]['physical_name']
                return remote_url, physical_name

        try:
            remote_url, physical_name = get_remote_url(f'https://sar-mpc.eu/api/v1/?product_type=AUX_CAL&validity_start={validity_start}&creation_date={creation_date}')
        except:
            remote_url, physical_name = get_remote_url(f'https://sar-mpc.eu/api/v1/?product_type=AUX_CAL&validity_start={validity_start}')

        download_file = os.path.join(self.aux_data_dir, physical_name)
        print(f'downloading {filename}.zip from {remote_url}')
        with requests.get(remote_url, stream=True) as r:
            with open(download_file, "wb") as f:
                f.write(r.content)

        with zipfile.ZipFile(download_file, 'r') as download_zip:
            download_zip.extractall(path=self.aux_data_dir)
        output_dir = f'{self.aux_data_dir}/{filename}'
        if not os.path.exists(output_dir):
            # in case the physical_name of the downloaded file does not exactly match the filename from manifest
            # the best found AUX-CAL file is copied to the filename from manifest
            shutil.copytree(download_file.rstrip('.zip'), output_dir)

    def get_noise_range_vectors(self, polarization):
        """ Get range noise from XML files and return noise, pixels and lines for non-zero elems"""
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        line = np.array(noiseRangeVector['line'])
        pixel = []
        noise = []

        for pix, n in zip(noiseRangeVector['pixel'], noiseRangeVector['noiseRangeLut']):
            n = np.array(n)
            n[n == 0] = np.nan
            noise.append(n)
            pixel.append(np.array(pix))
            
        if self.IPFversion <= 2.43:
            noise = [np.zeros_like(i) for i in noise]

        return line, pixel, noise
    
    def get_swath_id_vectors(self, polarization, line, pixel):
        swath_indices = [np.zeros(p.shape, int) for p in pixel]
        swathBounds = self.import_swathBounds(polarization)

        for iswath, swath_name in enumerate(swathBounds):
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((line >= fal) * (line <= lal))[0]
                for v1 in valid1:
                    valid2 = (pixel[v1] >= frs) * (pixel[v1] <= lrs)
                    swath_indices[v1][valid2] = iswath + 1
        return swath_indices

    def get_apg_vectors(self, polarization, line, pixel, min_size=3):
        """ Compute Antenna Pattern Gain APG=(1/EAP/RSL)**2 for each noise vectors """
        apg_vectors = [np.zeros(p.size) + np.nan for p in pixel]
        swath_indices = self.get_swath_id_vectors(polarization, line, pixel)
        
        for swid in self.swath_ids:
            swath_name = f'{self.obsMode}{swid}'
            eap_interpolator = self.get_eap_interpolator(swath_name, polarization)
            ba_interpolator = self.get_boresight_angle_interpolator(polarization)
            rsl_interpolator = self.get_range_spread_loss_interpolator(polarization)
            for i, (l, p, swids) in enumerate(zip(line, pixel, swath_indices)):
                gpi = np.where(swids == swid)[0]
                if gpi.size > min_size:
                    ba = ba_interpolator(l, p[gpi]).flatten()                    
                    eap = eap_interpolator(ba).flatten()
                    rsl = rsl_interpolator(l, p[gpi]).flatten()
                    apg = (1 / eap / rsl) ** 2
                    apg_vectors[i][gpi] = apg
            
        return apg_vectors

    def get_apg_scales_offsets(self):
        params = self.load_denoising_parameters_json()
        apg_id = f'{os.path.basename(self.filename)[:16]}_APG_{self.IPFversion:04.2f}'
        p = params[apg_id]
        offsets = []
        scales = []
        for i in range(5):
            offsets.append(p['B'][i*2] / p['Y_SCALE'])
            scales.append(p['B'][1+i*2] * p['A_SCALE'] / p['Y_SCALE'])
        return scales, offsets

    def get_swath_interpolator(self, polarization, swath_name, line, pixel, z):
        """ Prepare interpolators for one swath """
        swathBounds = self.import_swathBounds(polarization)
        swathBound = swathBounds[swath_name]
        swath_coords = (
            swathBound['firstAzimuthLine'],
            swathBound['lastAzimuthLine'],
            swathBound['firstRangeSample'],
            swathBound['lastRangeSample'],
        )
        pix_vec_fr = np.arange(min(swathBound['firstRangeSample']),
                               max(swathBound['lastRangeSample'])+1)

        z_vecs = []
        swath_lines = []
        for fal, lal, frs, lrs in zip(*swath_coords):
            valid1 = np.where((line >= fal) * (line <= lal))[0]
            if valid1.size == 0:
                continue
            for v1 in valid1:
                swath_lines.append(line[v1])
                valid2 = np.where(
                    (pixel[v1] >= frs) *
                    (pixel[v1] <= lrs) *
                    np.isfinite(z[v1]))[0]
                # interpolator for one line
                z_interp1 = InterpolatedUnivariateSpline(pixel[v1][valid2], z[v1][valid2])
                z_vecs.append(z_interp1(pix_vec_fr))
        # interpolator for one subswath
        z_interp2 = RectBivariateSpline(swath_lines, pix_vec_fr, np.array(z_vecs))
        return z_interp2, swath_coords
    
    def get_elevation_incidence_angle_interpolators(self, polarization):
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        xggp = np.unique(geolocationGridPoint['pixel'])
        yggp = np.unique(geolocationGridPoint['line'])
        elevation_map = np.array(geolocationGridPoint['elevationAngle']).reshape(len(yggp), len(xggp))
        incidence_map = np.array(geolocationGridPoint['incidenceAngle']).reshape(len(yggp), len(xggp))
        elevation_angle_interpolator = RectBivariateSpline(yggp, xggp, elevation_map)
        incidence_angle_interpolator = RectBivariateSpline(yggp, xggp, incidence_map)
        return elevation_angle_interpolator, incidence_angle_interpolator

    def get_elevation_incidence_angle_vectors(self, ea_interpolator, ia_interpolator, line, pixel):
        ea = [ea_interpolator(l, p).flatten() for (l, p) in zip(line, pixel)]
        ia = [ia_interpolator(l, p).flatten() for (l, p) in zip(line, pixel)]
        return ea, ia

    def get_calibration_vectors(self, polarization, line, pixel, name='sigmaNought'):
        swath_names = ['%s%s' % (self.obsMode, iSW) for iSW in self.swath_ids]
        calibrationVector = self.import_calibrationVector(polarization)
        s0line = np.array(calibrationVector['line'])
        s0pixel = np.array(calibrationVector['pixel'])
        sigma0 = np.array(calibrationVector[name])

        sigma0_vecs = [np.zeros_like(p_vec) + np.nan for p_vec in pixel]

        for swath_name in swath_names:
            sigma0interp, swath_coords = self.get_swath_interpolator(polarization, swath_name, s0line, s0pixel, sigma0)

            for fal, lal, frs, lrs in zip(*swath_coords):
                valid1 = np.where((line >= fal) * (line <= lal))[0]
                if valid1.size == 0:
                    continue
                for v1 in valid1:
                    valid2 = np.where(
                        (pixel[v1] >= frs) *
                        (pixel[v1] <= lrs))[0]
                    sigma0_vecs[v1][valid2] = sigma0interp(line[v1], pixel[v1][valid2])        
        return sigma0_vecs

    def get_noise_azimuth_vectors(self, polarization, line, pixel):
        """ Interpolate scalloping noise from XML files to input pixel/lines coords """
        scall = [np.zeros(p.size)+np.nan for p in pixel]
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        for iSW in self.swath_ids:
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfBlocks = len(noiseAzimuthVector[subswathID]['firstAzimuthLine'])
            for iBlk in range(numberOfBlocks):
                frs = noiseAzimuthVector[subswathID]['firstRangeSample'][iBlk]
                lrs = noiseAzimuthVector[subswathID]['lastRangeSample'][iBlk]
                fal = noiseAzimuthVector[subswathID]['firstAzimuthLine'][iBlk]
                lal = noiseAzimuthVector[subswathID]['lastAzimuthLine'][iBlk]
                y = np.array(noiseAzimuthVector[subswathID]['line'][iBlk])
                z = np.array(noiseAzimuthVector[subswathID]['noiseAzimuthLut'][iBlk])
                if y.size > 1:
                    nav_interp = InterpolatedUnivariateSpline(y, z, k=1)
                else:
                    nav_interp = lambda x: z

                line_gpi = np.where((line >= fal) * (line <= lal))[0]
                for line_i in line_gpi:
                    pixel_gpi = np.where((pixel[line_i] >= frs) * (pixel[line_i] <= lrs))[0]
                    scall[line_i][pixel_gpi] = nav_interp(line[line_i])
        return scall

    def calibrate_noise_vectors(self, noise, cal_s0, scall):
        """ Compute calibrated NESZ from input noise, sigma0 calibration and scalloping noise"""
        return [s * n / c**2 for n, c, s in zip(noise, cal_s0, scall)]

    def get_eap_interpolator(self, subswathID, polarization):
        """
        Prepare interpolator for Elevation Antenna Pattern.
        It computes EAP for input boresight angles

        """
        elevationAntennaPatternLUT = self.import_elevationAntennaPattern(polarization)
        eap_lut = np.array(elevationAntennaPatternLUT[subswathID]['elevationAntennaPattern'])
        recordLength = elevationAntennaPatternLUT[subswathID]['elevationAntennaPatternCount']
        eai_lut = elevationAntennaPatternLUT[subswathID]['elevationAngleIncrement']
        if recordLength == eap_lut.shape[0]:
            # in case if elevationAntennaPattern is given in dB
            amplitudeLUT = 10**(eap_lut/10)
        else:
            amplitudeLUT = np.sqrt(eap_lut[0:-1:2]**2 + eap_lut[1::2]**2)
            
        angleLUT = np.arange(-(recordLength//2),+(recordLength//2)+1) * eai_lut        
        eap_interpolator = InterpolatedUnivariateSpline(angleLUT, np.sqrt(amplitudeLUT))
        return eap_interpolator
    
    def get_boresight_angle_interpolator(self, polarization):
        """
        Prepare interpolator for boresaight angles.
        It computes BA for input x,y coordinates.

        """
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        xggp = np.unique(geolocationGridPoint['pixel'])
        yggp = np.unique(geolocationGridPoint['line'])
        elevation_map = np.array(geolocationGridPoint['elevationAngle']).reshape(len(yggp), len(xggp))

        antennaPattern = self.import_antennaPattern(polarization)
        relativeAzimuthTime = []
        for iSW in self.swath_ids:
            subswathID = '%s%s' % (self.obsMode, iSW)
            relativeAzimuthTime.append([ (t-self.time_coverage_center).total_seconds()
                                         for t in antennaPattern[subswathID]['azimuthTime'] ])
        relativeAzimuthTime = np.hstack(relativeAzimuthTime)
        sortIndex = np.argsort(relativeAzimuthTime)
        rollAngle = []
        for iSW in self.swath_ids:
            subswathID = '%s%s' % (self.obsMode, iSW)
            rollAngle.append(antennaPattern[subswathID]['roll'])
        relativeAzimuthTime = np.hstack(relativeAzimuthTime)
        rollAngle = np.hstack(rollAngle)
        rollAngleIntp = InterpolatedUnivariateSpline(relativeAzimuthTime[sortIndex], rollAngle[sortIndex])
        azimuthTime = [ (t-self.time_coverage_center).total_seconds() for t in geolocationGridPoint['azimuthTime'] ]
        azimuthTime = np.reshape(azimuthTime, (len(yggp), len(xggp)))
        roll_map = rollAngleIntp(azimuthTime)

        boresight_map = elevation_map - roll_map
        boresight_angle_interpolator = RectBivariateSpline(yggp, xggp, boresight_map)
        return boresight_angle_interpolator

    def get_range_spread_loss_interpolator(self, polarization):
        """ Prepare interpolator for Range Spreading Loss.
        It computes RSL for input x,y coordinates.

        """
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        xggp = np.unique(geolocationGridPoint['pixel'])
        yggp = np.unique(geolocationGridPoint['line'])
        referenceRange = float(self.annotationXML[polarization].getElementsByTagName(
                    'referenceRange')[0].childNodes[0].nodeValue)
        slantRangeTime = np.reshape(geolocationGridPoint['slantRangeTime'],(len(yggp),len(xggp)))
        rangeSpreadingLoss = (referenceRange / slantRangeTime / SPEED_OF_LIGHT * 2)**(3./2.)
        rsp_interpolator = RectBivariateSpline(yggp, xggp, rangeSpreadingLoss)
        return rsp_interpolator

    def get_shifted_noise_vectors(self, polarization, line, pixel, noise, skip = 4):
        """
        Estimate shift in range noise LUT relative to antenna gain pattern and correct for it.

        """
        noise_shifted = [np.zeros(p.size)+np.nan for p in pixel]
        swathBounds = self.import_swathBounds(polarization)
        # noise lut shift
        for swid in self.swath_ids:
            swath_name = f'{self.obsMode}{swid}'
            swathBound = swathBounds[swath_name]
            eap_interpolator = self.get_eap_interpolator(swath_name, polarization)
            ba_interpolator = self.get_boresight_angle_interpolator(polarization)
            rsp_interpolator = self.get_range_spread_loss_interpolator(polarization)
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((line >= fal) * (line <= lal))[0]
                for v1 in valid1:
                    valid_lin = line[v1]
                    # keep only pixels from that swath
                    valid2 = np.where(
                        (pixel[v1] >= frs) *
                        (pixel[v1] <= lrs) *
                        (np.isfinite(noise[v1])))[0]
                    # keep only unique pixels
                    valid_pix, valid_pix_i = np.unique(pixel[v1][valid2], return_index=True)
                    valid2 = valid2[valid_pix_i]

                    ba = ba_interpolator(valid_lin, valid_pix).flatten()
                    eap = eap_interpolator(ba).flatten()
                    rsp = rsp_interpolator(valid_lin, valid_pix).flatten()
                    apg = (1/eap/rsp)**2

                    noise_valid = np.array(noise[v1][valid2])
                    if np.allclose(noise_valid, noise_valid[0]):
                        noise_shifted[v1][valid2] = noise_valid
                    else:
                        noise_interpolator = InterpolatedUnivariateSpline(valid_pix, noise_valid)
                        pixel_shift = minimize(cost, 0, args=(valid_pix[skip:-skip], noise_interpolator, apg[skip:-skip])).x[0]
                        noise_shifted0 = noise_interpolator(valid_pix + pixel_shift)
                        noise_shifted[v1][valid2] = noise_shifted0

        return noise_shifted

    def get_corrected_noise_vectors(self, polarization, line, pixel, nesz, add_pb=True):
        """ Load scaling and offset coefficients from files and apply to input  NESZ """
        nesz_corrected = [np.zeros(p.size)+np.nan for p in pixel]
        swathBounds = self.import_swathBounds(polarization)
        ns, pb = self.import_denoisingCoefficients(polarization)[:2]
        for swid in self.swath_ids:
            swath_name = f'{self.obsMode}{swid}'
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((line >= fal) * (line <= lal))[0]
                for v1 in valid1:
                    valid2 = np.where((pixel[v1] >= frs) * (pixel[v1] <= lrs))[0]
                    nesz_corrected[v1][valid2] = nesz[v1][valid2] * ns[swath_name]
                    if add_pb:
                        nesz_corrected[v1][valid2] += pb[swath_name]
        return nesz_corrected

    def get_raw_sigma0_vectors(self, polarization, line, pixel, cal_s0, average_lines=111):
        """ Read DN_ values from input GeoTIff for a given lines, average in azimuth direction,
        compute sigma0, and return sigma0 for given pixels

        """
        ws2 = np.floor(average_lines / 2)
        raw_sigma0 = [np.zeros(p.size)+np.nan for p in pixel]
        src_filename = self.bands()[self.get_band_number(f'DN_{polarization}')]['SourceFilename']
        ds = gdal.Open(src_filename)
        img_data = ds.ReadAsArray().astype(float)
        img_data[img_data == 0] = np.nan
        for i in range(line.shape[0]):
            y0 = max(0, line[i]-ws2)
            y1 = min(ds.RasterYSize, line[i]+ws2)
            line_data = img_data[int(y0):int(y1)]
            dn_mean = np.nanmean(line_data, axis=0)
            raw_sigma0[i] = dn_mean[pixel[i]]**2 / cal_s0[i]**2
        return raw_sigma0

    def get_raw_sigma0_vectors_from_full_size(self, polarization, line, pixel, sigma0_fs, average_lines=111):
        """ Read DN_ values from input GeoTIff for a given lines, average in azimuth direction,
        compute sigma0, and return sigma0 for given pixels

        """
        ws2 = np.floor(average_lines / 2)
        raw_sigma0 = [np.zeros(p.size)+np.nan for p in pixel]
        for i in range(line.shape[0]):
            y0 = max(0, line[i]-ws2)
            y1 = min(sigma0_fs.shape[0], line[i]+ws2)
            raw_sigma0[i] = np.nanmean(sigma0_fs[int(y0):int(y1)], axis=0)[pixel[i]]
        return raw_sigma0

    def get_lons_vectors_from_full_size(self, line, pixel, lons_fs):
        """ Read lons from input GeoTIff for given lines and for given pixels
            from full size longitude matrix

        """
        lons_res = [np.zeros(p.size)+np.nan for p in pixel]
        for i in range(line.shape[0]):
            lons_res[i] = lons_fs[line[i]][pixel[i]]
        return lons_res

    def get_lats_vectors_from_full_size(self, line, pixel, lats_fs):
        """ Read lats from input GeoTIff for given lines and for given pixels
            from full size latitude matrix

        """
        lats_res = [np.zeros(p.size)+np.nan for p in pixel]
        for i in range(line.shape[0]):
            lats_res[i] = lats_fs[line[i]][pixel[i]]
        return lats_res

    def get_ia_vectors_from_full_size(self, line, pixel, ia_fs):
        """ Read incidence angle values from input GeoTIff for given lines and for given pixels
            from full size incidence angle matrix

        """
        ia_res = [np.zeros(p.size)+np.nan for p in pixel]
        for i in range(line.shape[0]):
            ia_res[i] = ia_fs[line[i]][pixel[i]]
        return ia_res

    def compute_rqm(self, s0, polarization, line, pixel, num_px=100, **kwargs):
        """ Compute Range Quality Metric from the input sigma0 """
        swathBounds = self.import_swathBounds(polarization)
        q_all = {}
        for swid in self.swath_ids[:-1]:
            q_subswath = []
            swath_name = f'{self.obsMode}{swid}'
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((line >= fal) * (line <= lal))[0]
                for v1 in valid1:
                    valid2a = np.where((pixel[v1] >= lrs-num_px) * (pixel[v1] <= lrs))[0]
                    valid2b = np.where((pixel[v1] >= lrs+1) * (pixel[v1] <= lrs+num_px+1))[0]
                    s0a = s0[v1][valid2a]
                    s0b = s0[v1][valid2b]
                    s0am = np.nanmean(s0a)
                    s0bm = np.nanmean(s0b)
                    s0as = np.nanstd(s0a)
                    s0bs = np.nanstd(s0a)
                    q = np.abs(s0am - s0bm) / (s0as + s0bs)
                    q_subswath.append([q, s0am, s0bm, s0as, s0bs, line[v1]])
            q_all[swath_name] = np.array(q_subswath)
        return q_all

    def get_range_quality_metric(self, polarization='HV', **kwargs):
        """ Compute sigma0 with three methods (ESA, SHIFTED, NERSC), compute RQM for each sigma0 """
        line, pixel, noise = self.get_noise_range_vectors(polarization)
        cal_s0 = self.get_calibration_vectors(polarization, line, pixel)
        scall = self.get_noise_azimuth_vectors(polarization, line, pixel)
        nesz = self.calibrate_noise_vectors(noise, cal_s0, scall)
        noise_shifted = self.get_shifted_noise_vectors(polarization, line, pixel, noise)
        nesz_shifted = self.calibrate_noise_vectors(noise_shifted, cal_s0, scall)
        nesz_corrected = self.get_corrected_noise_vectors(polarization, line, pixel, nesz_shifted)
        sigma0 = self.get_raw_sigma0_vectors(polarization, line, pixel, cal_s0)
        s0_esa   = [s0 - n0 for (s0,n0) in zip(sigma0, nesz)]
        s0_shift = [s0 - n0 for (s0,n0) in zip(sigma0, nesz_shifted)]
        s0_nersc = [s0 - n0 for (s0,n0) in zip(sigma0, nesz_corrected)]
        q = [self.compute_rqm(s0, polarization, line, pixel, **kwargs) for s0 in [s0_esa, s0_shift, s0_nersc]]

        alg_names = ['ESA', 'SHIFT', 'NERSC']
        var_names = ['RQM', 'AVG1', 'AVG2', 'STD1', 'STD2']
        q_all = {}
        for swid in self.swath_ids[:-1]:
            swath_name = f'{self.obsMode}{swid}'
            for alg_i, alg_name in enumerate(alg_names):
                for var_i, var_name in enumerate(var_names):
                    q_all[f'{var_name}_{swath_name}_{alg_name}'] = list(q[alg_i][swath_name][:, var_i])
            q_all[f'LINE_{swath_name}'] = list(q[alg_i][swath_name][:, 5])
        return q_all

    def experiment_get_data(self, polarization, average_lines, zoom_step):
        """ Prepare data for  noiseScaling and powerBalancing experiments """
        crop = {'IW':400, 'EW':200}[self.obsMode]
        swathBounds = self.import_swathBounds(polarization)
        line, pixel0, noise0 = self.get_noise_range_vectors(polarization)
        cal_s00 = self.get_calibration_vectors(polarization, line, pixel0)
        # zoom:
        pixel = [np.arange(p[0], p[-1], zoom_step) for p in pixel0]
        noise = [interp1d(p, n)(p2) for (p,n,p2) in zip(pixel0, noise0, pixel)]
        cal_s0 = [interp1d(p, n)(p2) for (p,n,p2) in zip(pixel0, cal_s00, pixel)]

        noise_shifted = self.get_shifted_noise_vectors(polarization, line, pixel, noise)
        scall = self.get_noise_azimuth_vectors(polarization, line, pixel)
        nesz = self.calibrate_noise_vectors(noise_shifted, cal_s0, scall)
        sigma0_fs = self.get_raw_sigma0_full_size(polarization)
        sigma0 = self.get_raw_sigma0_vectors_from_full_size(
            polarization, line, pixel, sigma0_fs, average_lines=average_lines)

        return line, pixel, sigma0, nesz, crop, swathBounds

    def experiment_noiseScaling(self, polarization, average_lines=777, zoom_step=2):
        """ Compute noise scaling coefficients for each range noise line and save as NPZ """
        line, pixel, sigma0, nesz, crop, swathBounds = self.experiment_get_data(
            polarization, average_lines, zoom_step)

        results = {}
        results['IPFversion'] = self.IPFversion
        for swid in self.swath_ids:
            swath_name = f'{self.obsMode}{swid}'
            results[swath_name] = {
                'sigma0':[],
                'noiseEquivalentSigma0':[],
                'scalingFactor':[],
                'correlationCoefficient':[],
                'fitResidual':[] }
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where(
                    (line >= fal) *
                    (line <= lal) *
                    (line > (average_lines / 2)) *
                    (line < (line[-1] - average_lines / 2)))[0]
                for v1 in valid1:
                    valid2 = np.where(
                        (pixel[v1] >= frs+crop) *
                        (pixel[v1] <= lrs-crop) *
                        np.isfinite(nesz[v1]))[0]
                    meanS0 = sigma0[v1][valid2]
                    meanN0 = nesz[v1][valid2]
                    pixelIndex = pixel[v1][valid2]
                    (scalingFactor,
                     correlationCoefficient,
                     fitResidual) = fit_noise_scaling_coeff(meanS0, meanN0, pixelIndex)
                    results[swath_name]['sigma0'].append(meanS0)
                    results[swath_name]['noiseEquivalentSigma0'].append(meanN0)
                    results[swath_name]['scalingFactor'].append(scalingFactor)
                    results[swath_name]['correlationCoefficient'].append(correlationCoefficient)
                    results[swath_name]['fitResidual'].append(fitResidual)
        np.savez(self.filename.split('.')[0] + '_noiseScaling.npz', **results)

    def experiment_powerBalancing(self, polarization, average_lines=777, zoom_step=2):
        """ Compute power balancing coefficients for each range noise line and save as NPZ """
        line, pixel, sigma0, nesz, crop, swathBounds = self.experiment_get_data(
            polarization, average_lines, zoom_step)
        nesz_corrected = self.get_corrected_noise_vectors(
            polarization, line, pixel, nesz, add_pb=False)

        num_swaths = len(self.swath_ids)
        swath_names = ['%s%s' % (self.obsMode, iSW) for iSW in self.swath_ids]

        results = {}
        results['IPFversion'] = self.IPFversion
        tmp_results = {}
        for swath_name in swath_names:
            results[swath_name] = {
                'sigma0':[],
                'noiseEquivalentSigma0':[],
                'correlationCoefficient':[],
                'fitResidual':[],
                'balancingPower': []}
            tmp_results[swath_name] = {}

        valid_lines = np.where(
            (line > (average_lines / 2)) *
            (line < (line[-1] - average_lines / 2)))[0]
        for li in valid_lines:
            # find frs, lrs for all swaths at this line
            frs = {}
            lrs = {}
            for swath_name in swath_names:
                swathBound = swathBounds[swath_name]
                zipped = zip(
                    swathBound['firstAzimuthLine'],
                    swathBound['lastAzimuthLine'],
                    swathBound['firstRangeSample'],
                    swathBound['lastRangeSample'],
                )
                for fal, lal, frstmp, lrstmp in zipped:
                    if line[li] >= fal and line[li] <= lal:
                        frs[swath_name] = frstmp
                        lrs[swath_name] = lrstmp
                        break

            if swath_names != sorted(list(frs.keys())):
                continue

            blockN0 = np.zeros(nesz[li].shape) + np.nan
            blockRN0 = np.zeros(nesz[li].shape) + np.nan
            valid2_zero_size = False
            fitCoefficients = []
            for swath_name in swath_names:
                swathBound = swathBounds[swath_name]
                valid2 = np.where(
                    (pixel[li] >= frs[swath_name]+crop) *
                    (pixel[li] <= lrs[swath_name]-crop) *
                    np.isfinite(nesz[li]))[0]
                if valid2.size == 0:
                    valid2_zero_size = True
                    break
                meanS0 = sigma0[li][valid2]
                meanN0 = nesz_corrected[li][valid2]
                blockN0[valid2] = nesz_corrected[li][valid2]
                meanRN0 = nesz[li][valid2]
                blockRN0[valid2] = nesz[li][valid2]
                pixelIndex = pixel[li][valid2]
                fitResults = np.polyfit(pixelIndex, meanS0 - meanN0, deg=1, full=True)
                fitCoefficients.append(fitResults[0])
                tmp_results[swath_name]['sigma0'] = meanS0
                tmp_results[swath_name]['noiseEquivalentSigma0'] = meanRN0
                tmp_results[swath_name]['correlationCoefficient'] = np.corrcoef(meanS0, meanN0)[0,1]
                tmp_results[swath_name]['fitResidual'] = fitResults[1].item()

            if valid2_zero_size:
                continue

            if np.any(np.isnan(fitCoefficients)):
                continue

            balancingPower = np.zeros(num_swaths)
            for i in range(num_swaths - 1):
                swath_name = f'{self.obsMode}{i+1}'
                swathBound = swathBounds[swath_name]
                power1 = fitCoefficients[i][0] * lrs[swath_name] + fitCoefficients[i][1]
                # Compute power right to a boundary as slope*interswathBounds + residual coef.
                power2 = fitCoefficients[i+1][0] * lrs[swath_name] + fitCoefficients[i+1][1]
                balancingPower[i+1] = power2 - power1
            balancingPower = np.cumsum(balancingPower)

            for iSW, swath_name in zip(self.swath_ids, swath_names):
                swathBound = swathBounds[swath_name]
                valid2 = np.where(
                    (pixel[li] >= frs[swath_name]+crop) *
                    (pixel[li] <= lrs[swath_name]-crop) *
                    np.isfinite(nesz[li]))[0]
                blockN0[valid2] += balancingPower[iSW-1]

            valid3 = (pixel[li] >= frs[f'{self.obsMode}2'] + crop)
            powerBias = np.nanmean((blockRN0-blockN0)[valid3])
            balancingPower += powerBias

            for iSW, swath_name in zip(self.swath_ids, swath_names):
                tmp_results[swath_name]['balancingPower'] = balancingPower[iSW-1]

            for swath_name in swath_names:
                for key in tmp_results[swath_name]:
                    results[swath_name][key].append(tmp_results[swath_name][key])

        np.savez(self.filename.split('.')[0] + '_powerBalancing.npz', **results)

    def get_scalloping_full_size(self, polarization):
        """ Interpolate noise azimuth vector to full resolution for all blocks """
        scall_fs = np.zeros(self.shape()) + np.nan
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        swath_names = ['%s%s' % (self.obsMode, iSW) for iSW in self.swath_ids]
        for swath_name in swath_names:
            nav = noiseAzimuthVector[swath_name]
            zipped = zip(
                nav['firstAzimuthLine'],
                nav['lastAzimuthLine'],
                nav['firstRangeSample'],
                nav['lastRangeSample'],
                nav['line'],
                nav['noiseAzimuthLut'],
            )
            for fal, lal, frs, lrs, y, z in zipped:
                if isinstance(y, list):
                    nav_interp = InterpolatedUnivariateSpline(y, z, k=1)
                else:
                    nav_interp = lambda x: z
                lin_vec_fr = np.arange(fal, lal+1)
                z_vec_fr = nav_interp(lin_vec_fr)
                z_arr = np.repeat([z_vec_fr], (lrs-frs+1), axis=0).T
                scall_fs[fal:lal+1, frs:lrs+1] = z_arr
        return scall_fs

    def interp_nrv_full_size(self, z, line, pixel, polarization, power=1):
        """ Interpolate noise range vectors to full size """
        z_fs = np.zeros(self.shape()) + np.nan
        swath_names = ['%s%s' % (self.obsMode, iSW) for iSW in self.swath_ids]
        for swath_name in swath_names:
            z_interp2, swath_coords = self.get_swath_interpolator(
                polarization, swath_name, line, pixel, z)
            for fal, lal, frs, lrs in zip(*swath_coords):
                pix_vec_fr = np.arange(frs, lrs+1)
                lin_vec_fr = np.arange(fal, lal+1)
                z_arr_fs = z_interp2(lin_vec_fr, pix_vec_fr)
                if power != 1:
                    z_arr_fs = z_arr_fs**power
                z_fs[fal:lal+1, frs:lrs+1] = z_arr_fs
        return z_fs

    def get_calibration_full_size(self, polarization, power=1):
        """ Get calibration constant on full size matrix """
        calibrationVector = self.import_calibrationVector(polarization)
        line = np.array(calibrationVector['line'])
        pixel = np.array(calibrationVector['pixel'])
        s0 = np.array(calibrationVector['sigmaNought'])
        return self.interp_nrv_full_size(s0, line, pixel, polarization, power=power)

    def get_corrected_nesz_full_size(self, polarization, nesz):
        """ Get corrected NESZ on full size matrix """
        nesz_corrected = np.array(nesz)
        swathBounds = self.import_swathBounds(polarization)
        ns, pb = self.import_denoisingCoefficients(polarization)[:2]
        for swid in self.swath_ids:
            swath_name = f'{self.obsMode}{swid}'
            # skip correction id NS/PB coeffs are not available (e.g. HH or VV)
            if swath_name not in ns:
                continue
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                nesz_corrected[fal:lal+1, frs:lrs+1] *= ns[swath_name]
                nesz_corrected[fal:lal+1, frs:lrs+1] += pb[swath_name]
        return nesz_corrected

    def get_raw_sigma0_full_size(self, polarization, min_dn=0):
        """ Read DN from input GeoTiff file and calibrate """
        src_filename = self.bands()[self.get_band_number(f'DN_{polarization}')]['SourceFilename']
        ds = gdal.Open(src_filename)
        dn = ds.ReadAsArray()

        sigma0_fs = dn.astype(float)**2 / self.get_calibration_full_size(polarization, power=2)
        sigma0_fs[dn <= min_dn] = np.nan
        return sigma0_fs

    def export_noise_xml(self, polarization, output_path):
        """ Export corrected (shifted and scaled) range noise into XML file """
        crosspol_noise_file = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*')
                          if 'noise-s1' in fn and '-%s-' % polarization.lower() in fn][0]

        line, pixel, noise0 = self.get_noise_range_vectors(polarization)
        noise1 = self.get_shifted_noise_vectors(polarization, line, pixel, noise0)
        noise2 = self.get_corrected_noise_vectors(polarization, line, pixel, noise1)
        tree = ET.parse(crosspol_noise_file)
        root = tree.getroot()
        for noiseRangeVector, pixel_vec, noise_vec in zip(root.iter('noiseRangeVector'), pixel, noise2):
            noise_vec[np.isnan(noise_vec)] = 0
            noiseRangeVector.find('pixel').text = ' '.join([f'{p}' for p in list(pixel_vec)])
            noiseRangeVector.find('noiseRangeLut').text = ' '.join([f'{p}' for p in list(noise_vec)])
        tree.write(os.path.join(output_path, os.path.basename(crosspol_noise_file)))
        return crosspol_noise_file

    def get_noise_apg_vectors(self, polarization, line, pixel):
        noise = [np.zeros_like(i) for i in pixel]
        scales, offsets = self.get_apg_scales_offsets()
        apg = self.get_apg_vectors(polarization, line, pixel)
        swath_ids = self.get_swath_id_vectors(polarization, line, pixel)
        for i in range(len(apg)):
            for j in range(1,6):
                gpi = swath_ids[i] == j
                noise[i][gpi] = offsets[j-1] + apg[i][gpi] * scales[j-1]
        return noise

    def get_nesz_full_size(self, polarization, algorithm):
        # ESA noise vectors
        line, pixel, noise = self.get_noise_range_vectors(polarization)
        if algorithm == 'NERSC':
            # NERSC correction of noise shift in range direction
            noise = self.get_shifted_noise_vectors(polarization, line, pixel, noise)
        elif algorithm == 'NERSC_APG':
            # APG-based noise vectors
            noise = self.get_noise_apg_vectors(polarization, line, pixel)
        # noise calibration
        cal_s0 = self.get_calibration_vectors(polarization, line, pixel)
        nesz = [n / c**2 for (n,c) in zip(noise, cal_s0)]
        # noise full size
        nesz_fs = self.interp_nrv_full_size(nesz, line, pixel, polarization)
        # scalloping correction
        nesz_fs *= self.get_scalloping_full_size(polarization)
        # NERSC correction of NESZ magnitude
        if algorithm == 'NERSC':
            nesz_fs = self.get_corrected_nesz_full_size(polarization, nesz_fs)
        return nesz_fs

    def remove_thermal_noise(self, polarization, algorithm='NERSC', remove_negative=True, min_dn=0):
        nesz_fs = self.get_nesz_full_size(polarization, algorithm)
        sigma0 = self.get_raw_sigma0_full_size(polarization, min_dn=min_dn)
        sigma0 -= nesz_fs

        if remove_negative:
            sigma0 = fill_gaps(sigma0, sigma0 <= 0)
        return sigma0

    def import_noiseVector(self, polarization):
        ''' Import noise vectors from noise annotation XML DOM '''
        noiseRangeVector = { 'azimuthTime':[],
                             'line':[],
                             'pixel':[],
                             'noiseRangeLut':[] }
        noiseAzimuthVector = { '%s%s' % (self.obsMode, li):
                                   { 'firstAzimuthLine':[],
                                     'firstRangeSample':[],
                                     'lastAzimuthLine':[],
                                     'lastRangeSample':[],
                                     'line':[],
                                     'noiseAzimuthLut':[] }
                               for li in self.swath_ids }
        # ESA changed the noise vector structure from IPFv 2.9 to include azimuth variation
        if self.IPFversion < 2.9:
            noiseVectorList = self.noiseXML[polarization].getElementsByTagName('noiseVector')
            for iList in noiseVectorList:
                noiseRangeVector['azimuthTime'].append(
                    datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']),
                                      '%Y-%m-%dT%H:%M:%S.%f'))
                noiseRangeVector['line'].append(
                    get_DOM_nodeValue(iList,['line'],'int'))
                noiseRangeVector['pixel'].append(
                    get_DOM_nodeValue(iList,['pixel'],'int'))
                # To keep consistency, noiseLut is stored as noiseRangeLut
                noiseRangeVector['noiseRangeLut'].append(
                    get_DOM_nodeValue(iList,['noiseLut'],'float'))
            for iSW in self.swath_ids:
                swath = self.obsMode + str(iSW)
                noiseAzimuthVector[swath]['firstAzimuthLine'].append(0)
                noiseAzimuthVector[swath]['firstRangeSample'].append(0)
                noiseAzimuthVector[swath]['lastAzimuthLine'].append(self.shape()[0]-1)
                noiseAzimuthVector[swath]['lastRangeSample'].append(self.shape()[1]-1)
                noiseAzimuthVector[swath]['line'].append([0, self.shape()[0]-1])
                # noiseAzimuthLut is filled with a vector of ones
                noiseAzimuthVector[swath]['noiseAzimuthLut'].append([1.0, 1.0])
        elif self.IPFversion >= 2.9:
            noiseRangeVectorList = self.noiseXML[polarization].getElementsByTagName(
                'noiseRangeVector')
            for iList in noiseRangeVectorList:
                noiseRangeVector['azimuthTime'].append(
                    datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']),
                                      '%Y-%m-%dT%H:%M:%S.%f'))
                noiseRangeVector['line'].append(
                    get_DOM_nodeValue(iList,['line'],'int'))
                noiseRangeVector['pixel'].append(
                    get_DOM_nodeValue(iList,['pixel'],'int'))
                noiseRangeVector['noiseRangeLut'].append(
                    get_DOM_nodeValue(iList,['noiseRangeLut'],'float'))
            noiseAzimuthVectorList = self.noiseXML[polarization].getElementsByTagName(
                'noiseAzimuthVector')
            for iList in noiseAzimuthVectorList:
                swath = get_DOM_nodeValue(iList,['swath'],'str')
                for k in noiseAzimuthVector[swath].keys():
                    if k=='noiseAzimuthLut':
                        noiseAzimuthVector[swath][k].append(
                            get_DOM_nodeValue(iList,[k],'float'))
                    else:
                        noiseAzimuthVector[swath][k].append(
                            get_DOM_nodeValue(iList,[k],'int'))
        return noiseRangeVector, noiseAzimuthVector

    def import_calibrationVector(self, polarization):
        ''' Import calibration vectors from calibration annotation XML DOM '''
        calibrationVectorList = self.calibrationXML[polarization].getElementsByTagName(
            'calibrationVector')
        calibrationVector = { 'azimuthTime':[],
                              'line':[],
                              'pixel':[],
                              'sigmaNought':[],
                              'betaNought':[],
                              'gamma':[],
                              'dn':[] }
        for iList in calibrationVectorList:
            for k in calibrationVector.keys():
                if k=='azimuthTime':
                    calibrationVector[k].append(
                        datetime.strptime(get_DOM_nodeValue(iList,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                elif k in ['line', 'pixel']:
                    calibrationVector[k].append(
                        get_DOM_nodeValue(iList,[k],'int'))
                else:
                    calibrationVector[k].append(
                        get_DOM_nodeValue(iList,[k],'float'))
        return calibrationVector

    def import_swathBounds(self, polarization):
        ''' Import swath bounds information from annotation XML DOM '''
        swathMergeList = self.annotationXML[polarization].getElementsByTagName('swathMerge')
        swathBounds = { '%s%s' % (self.obsMode, li):
                            { 'azimuthTime':[],
                              'firstAzimuthLine':[],
                              'firstRangeSample':[],
                              'lastAzimuthLine':[],
                              'lastRangeSample':[] }
                        for li in self.swath_ids }
        for iList1 in swathMergeList:
            swath = get_DOM_nodeValue(iList1,['swath'])
            swathBoundsList = iList1.getElementsByTagName('swathBounds')
            for iList2 in swathBoundsList:
                for k in swathBounds[swath].keys():
                    if k=='azimuthTime':
                        swathBounds[swath][k].append(
                            datetime.strptime(get_DOM_nodeValue(iList2,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                    else:
                        swathBounds[swath][k].append(get_DOM_nodeValue(iList2,[k],'int'))
        return swathBounds

    def import_elevationAntennaPattern(self, polarization):
        ''' Import elevation antenna pattern from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        elevationAntennaPattern = { '%s%s' % (self.obsMode, li):
                                        { 'elevationAngleIncrement':[],
                                          'elevationAntennaPattern':[],
                                          'absoluteCalibrationConstant':[],
                                          'noiseCalibrationFactor':[],
                                          'elevationAntennaPatternCount': None
                                        }
                                    for li in self.swath_ids }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            pol = get_DOM_nodeValue(iList,['polarisation'])
            if (swath in elevationAntennaPattern.keys()) and (pol==polarization):
                elem = iList.getElementsByTagName('elevationAntennaPattern')[0]
                for k in elevationAntennaPattern[swath].keys():
                    if k=='elevationAngleIncrement':
                        elevationAntennaPattern[swath][k] = get_DOM_nodeValue(elem,[k],'float')
                    elif k=='elevationAntennaPattern':
                        elevationAntennaPattern[swath][k] = get_DOM_nodeValue(elem,['values'],'float')
                        elevationAntennaPattern[swath]['elevationAntennaPatternCount'] = int(elem.getElementsByTagName('values')[0].getAttribute('count'))
                    elif k in ['absoluteCalibrationConstant', 'noiseCalibrationFactor']:
                        elevationAntennaPattern[swath][k] = get_DOM_nodeValue(iList,[k],'float')
        return elevationAntennaPattern

    def import_geolocationGridPoint(self, polarization):
        ''' Import geolocation grid point from annotation XML DOM '''
        geolocationGridPointList = self.annotationXML[polarization].getElementsByTagName(
            'geolocationGridPoint')
        geolocationGridPoint = { 'azimuthTime':[],
                                 'slantRangeTime':[],
                                 'line':[],
                                 'pixel':[],
                                 'latitude':[],
                                 'longitude':[],
                                 'height':[],
                                 'incidenceAngle':[],
                                 'elevationAngle':[] }
        for iList in geolocationGridPointList:
            for k in geolocationGridPoint.keys():
                if k=='azimuthTime':
                    geolocationGridPoint[k].append(
                        datetime.strptime(get_DOM_nodeValue(iList,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                elif k in ['line', 'pixel']:
                    geolocationGridPoint[k].append(
                        get_DOM_nodeValue(iList,[k],'int'))
                else:
                    geolocationGridPoint[k].append(
                        get_DOM_nodeValue(iList,[k],'float'))
        return geolocationGridPoint

    def import_antennaPattern(self, polarization, teta_ref=29.45):
        ''' Import antenna pattern from annotation XML DOM '''
        antennaPatternList = self.annotationXML[polarization].getElementsByTagName('antennaPattern')
        antennaPatternList = antennaPatternList[1:]
        antennaPattern = { '%s%s' % (self.obsMode, li):
                               { 'azimuthTime':[],
                                 'slantRangeTime':[],
                                 'elevationAngle':[],
                                 'elevationPattern':[],
                                 'incidenceAngle':[],
                                 'terrainHeight':[],
                                 'roll':[] }
                           for li in self.swath_ids }
        for iList in antennaPatternList:
            swath = get_DOM_nodeValue(iList,['swath'])
            for k in antennaPattern[swath].keys():
                if k=='azimuthTime':
                    antennaPattern[swath][k].append(
                        datetime.strptime(get_DOM_nodeValue(iList,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                elif k == 'roll' and self.IPFversion <= 2.36:
                    # Sentinel-1-Level-1-Detailed-Algorithm-Definition.pdf, p. 9-12, eq. 9-19
                    antennaPattern[swath][k].append(teta_ref)
                else:
                    antennaPattern[swath][k].append(get_DOM_nodeValue(iList,[k],'float'))
        return antennaPattern

    def load_denoising_parameters_json(self):
        denoise_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'denoising_parameters.json')
        with open(denoise_filename) as f:
            params = json.load(f)
        return params

    def import_denoisingCoefficients(self, polarization, load_extra_scaling=False):
        ''' Import denoising coefficients '''
        filename_parts = os.path.basename(self.filename).split('_')
        platform = filename_parts[0]
        mode = filename_parts[1]
        resolution = filename_parts[2]
        params = self.load_denoising_parameters_json()

        noiseScalingParameters = {}
        powerBalancingParameters = {}
        extraScalingParameters = {}
        noiseVarianceParameters = {}
        IPFversion = float(self.IPFversion)
        sensingDate = datetime.strptime(self.filename.split(os.sep)[-1].split('_')[4], '%Y%m%dT%H%M%S')
        if platform=='S1B' and IPFversion==2.72 and sensingDate >= datetime(2017,1,16,13,42,34):
            # Adaption for special case.
            # ESA abrubtly changed scaling LUT in AUX_PP1 from 20170116 while keeping the IPFv.
            # After this change, the scaling parameters seems be much closer to those of IPFv 2.8.
            IPFversion = 2.8

        base_key = f'{platform}_{mode}_{resolution}_{polarization}'
        for iSW in self.swath_ids:
            subswathID = '%s%s' % (self.obsMode, iSW)

            ns_key = f'{base_key}_NS_%0.1f' % IPFversion
            if ns_key in params:
                noiseScalingParameters[subswathID] = params[ns_key].get(subswathID, 1)
            else:
                print(f'WARNING: noise scaling for {subswathID} (IPF:{IPFversion}) is missing.')
                noiseScalingParameters[subswathID] = 1

            pb_key = f'{base_key}_PB_%0.1f' % IPFversion
            if pb_key in params:
                powerBalancingParameters[subswathID] = params[pb_key].get(subswathID, 0)
            else:
                print(f'WARNING: power balancing for {subswathID} (IPF:{IPFversion}) is missing.')
                powerBalancingParameters[subswathID] = 0

            if not load_extra_scaling:
                continue
            es_key = f'{base_key}_ES_%0.1f' % IPFversion
            if es_key in params:
                extraScalingParameters[subswathID] = params[es_key][subswathID]
                extraScalingParameters['SNNR'] = params[es_key]['SNNR']
            else:
                print(f'WARNING: extra scaling for {subswathID} (IPF:{IPFversion}) is missing.')
                extraScalingParameters['SNNR'] = np.linspace(-30,+30,601)
                extraScalingParameters[subswathID] = np.ones(601)

            nv_key = f'{base_key}_NV_%0.1f' % IPFversion
            if pb_key in params:
                nv_key[subswathID] = params[nv_key].get(subswathID, 0)
            else:
                print(f'WARNING: noise variance for {subswathID} (IPF:{IPFversion}) is missing.')

        return ( noiseScalingParameters, powerBalancingParameters, extraScalingParameters,
                 noiseVarianceParameters )

    def remove_texture_noise(self, polarization, window=3, weight=0.15, s0_min=0, remove_negative=True, algorithm='NERSC', min_dn=0, **kwargs):
        """ Thermal noise removal followed by textural noise compensation using Method2

        Method2 is implemented as a weighted average of sigma0 and sigma0 smoothed with
        a gaussian filter. Weight of sigma0 is proportional to SNR. Total noise power
        is preserved by ofsetting the output signal by mean noise. Values below <s0_min>
        are clipped to s0_min.

        Parameters
        ----------
        polarization : str
            'HH' or 'HV'
        window : int
            Size of window in the gaussian filter
        weight : float
            Weight of smoothed signal
        s0_min : float
            Minimum value of sigma0 for clipping

        Returns
        -------
        sigma0 : 2d numpy.ndarray
            Full size array with thermal and texture noise removed

        """

        if self.IPFversion == 3.2:
            self.IPFversion = 3.1
        sigma0 = self.get_raw_sigma0_full_size(polarization, min_dn=min_dn)
        nesz = self.get_nesz_full_size(polarization, algorithm)
        sigma0 -= nesz

        s0_offset = np.nanmean(nesz)
        sigma0g = gaussian_filter(sigma0, window)
        snr = sigma0g / nesz
        sigma0o = (weight * sigma0g + snr * sigma0) / (weight + snr) + s0_offset

        if remove_negative:
            sigma0o = fill_gaps(sigma0o, sigma0o <= s0_min)

        return sigma0o
