import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
from scipy.stats import pearsonr

from s1denoise.S1_TOPS_GRD_NoiseCorrection import Sentinel1Image, fit_noise_scaling_coeff

SPEED_OF_LIGHT = 299792458.


def cost(x, pix_valid, interp, y_ref):
    """ Cost function for finding noise LUT shift in Range """
    y = interp(pix_valid+x)
    return 1 - pearsonr(y_ref, y)[0]


class Sentinel1CalVal(Sentinel1Image):
    """ Cal/Val routines for Sentinel-1 performed on range noise vector coordinatess"""

    def get_noise_range_vectors(self, polarization):
        """ Get range noise from XML files and return noise, pixels and lines for non-zero elems"""
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        nrv_line = np.array(noiseRangeVector['line'])
        nrv_pixel = []
        nrv_noise = []

        for pix, noise in zip(noiseRangeVector['pixel'], noiseRangeVector['noiseRangeLut']):
            noise = np.array(noise)
            gpi = np.where(noise > 0)[0]
            nrv_noise.append(noise[gpi])
            nrv_pixel.append(np.array(pix)[gpi])

        return nrv_line, nrv_pixel, nrv_noise

    def get_calibration_vectors(self, polarization, nrv_line, nrv_pixel):
        """ Interpolate sigma0 calibration from XML file to the input line/pixel coordinates """
        nrv_cal_s0 = [np.zeros(p.size)+np.nan for p in nrv_pixel]
        calibrationVector = self.import_calibrationVector(polarization)
        cal_s0 = np.array(calibrationVector['sigmaNought'])
        swathBounds = self.import_swathBounds(polarization)
        for iSW in self.swath_ids:
            swathBound = swathBounds['%s%s' % (self.obsMode, iSW)]
            nrv_line_gpi = ((nrv_line >= min(swathBound['firstAzimuthLine'])) *
                            (nrv_line <= max(swathBound['lastAzimuthLine'])))
            cal_line = np.array(calibrationVector['line'])
            cal_line_gpi = ((cal_line >= min(swathBound['firstAzimuthLine'])) *
                            (cal_line <= max(swathBound['lastAzimuthLine'])))
            cal_line = cal_line[cal_line_gpi]
            cal_pixel = np.array(calibrationVector['pixel'])[cal_line_gpi][0]
            cal_pixel_gpi = ((cal_pixel >= min(swathBound['firstRangeSample'])) *
                             (cal_pixel <= max(swathBound['lastRangeSample'])))
            cal_pixel = cal_pixel[cal_pixel_gpi]
            cal_s0_pixlin = cal_s0[cal_line_gpi][:, cal_pixel_gpi]
            cal_s0_interp = RectBivariateSpline(cal_line, cal_pixel, cal_s0_pixlin, kx=1, ky=1)
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                nrv_line_gpi = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for nrv_line_i in nrv_line_gpi:
                    nrv_pixel_gpi = np.where((nrv_pixel[nrv_line_i] >= frs) * (nrv_pixel[nrv_line_i] <= lrs))[0]
                    nrv_cal_s0[nrv_line_i][nrv_pixel_gpi] = cal_s0_interp(
                        nrv_line[nrv_line_i],
                        nrv_pixel[nrv_line_i][nrv_pixel_gpi])
        return nrv_cal_s0

    def get_noise_azimuth_vectors(self, polarization, nrv_line, nrv_pixel):
        """ Interpolate scalloping noise from XML files to input pixel/lines coords """
        nrv_scall = [np.zeros(p.size)+np.nan for p in nrv_pixel]
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        for iSW in self.swath_ids:
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfBlocks = len(noiseAzimuthVector[subswathID]['firstAzimuthLine'])
            for iBlk in range(numberOfBlocks):
                frs = noiseAzimuthVector[subswathID]['firstRangeSample'][iBlk]
                lrs = noiseAzimuthVector[subswathID]['lastRangeSample'][iBlk]
                fal = noiseAzimuthVector[subswathID]['firstAzimuthLine'][iBlk]
                lal = noiseAzimuthVector[subswathID]['lastAzimuthLine'][iBlk]
                y = noiseAzimuthVector[subswathID]['line'][iBlk]
                z = noiseAzimuthVector[subswathID]['noiseAzimuthLut'][iBlk]
                nav_interp = InterpolatedUnivariateSpline(y, z, k=1)

                nrv_line_gpi = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for nrv_line_i in nrv_line_gpi:
                    nrv_pixel_gpi = np.where((nrv_pixel[nrv_line_i] >= frs) * (nrv_pixel[nrv_line_i] <= lrs))[0]
                    nrv_scall[nrv_line_i][nrv_pixel_gpi] = nav_interp(nrv_line[nrv_line_i])
        return nrv_scall

    def get_raw_nesz_vectors(self, nrv_noise, nrv_cal_s0, nrv_scall):
        """ Compute calibrated raw NESZ """
        # calibrate noise and compute NESZ
        nrv_nesz = []
        for n, cal, scall in zip(nrv_noise, nrv_cal_s0, nrv_scall):
            n_calib = scall * n / cal**2
            nrv_nesz.append(n_calib)
        return nrv_nesz

    def get_eap_interpolator(self, subswathID, polarization):
        """
        Prepare interpolator for Elevation Antenna Pattern.
        It computes EAP for input boresight angles

        """
        elevationAntennaPatternLUT = self.import_elevationAntennaPattern(polarization)
        eap_lut = np.array(elevationAntennaPatternLUT[subswathID]['elevationAntennaPattern'])
        eai_lut = elevationAntennaPatternLUT[subswathID]['elevationAngleIncrement']
        recordLength = len(eap_lut)/2
        angleLUT = np.arange(-(recordLength//2),+(recordLength//2)+1) * eai_lut
        amplitudeLUT = np.sqrt(eap_lut[0::2]**2 + eap_lut[1::2]**2)
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
        rollAngleIntp = InterpolatedUnivariateSpline(
            relativeAzimuthTime[sortIndex], rollAngle[sortIndex])
        azimuthTime = [ (t-self.time_coverage_center).total_seconds()
                          for t in geolocationGridPoint['azimuthTime'] ]
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

    def get_shifted_nesz_vectors(self, polarization, nrv_line, nrv_pixel, nrv_noise, nrv_cal_s0, nrv_scall):
        """
        Estimate shift in range noise LUT relative to antenna gain pattern and correct for it.

        """
        nrv_nesz_shifted = [np.zeros(p.size)+np.nan for p in nrv_pixel]
        swathBounds = self.import_swathBounds(polarization)
        # noise lut shift
        for swid in self.swath_ids:
            swath_name = f'EW{swid}'
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
                valid1 = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for v1 in valid1:
                    valid_lin = nrv_line[v1]
                    valid2 = np.where((nrv_pixel[v1] >= frs) * (nrv_pixel[v1] <= lrs))[0]
                    valid_pix = nrv_pixel[v1][valid2]

                    ba = ba_interpolator(valid_lin, valid_pix).flatten()
                    eap = eap_interpolator(ba).flatten()
                    rsp = rsp_interpolator(valid_lin, valid_pix).flatten()
                    apg = (1/eap/rsp)**2

                    noise = np.array(nrv_noise[v1][valid2])
                    noise_interpolator = InterpolatedUnivariateSpline(valid_pix, noise)
                    skip = 4
                    pixel_shift = minimize(cost, 0, args=(valid_pix[skip:-skip], noise_interpolator, apg[skip:-skip])).x[0]

                    noise_shifted = noise_interpolator(valid_pix + pixel_shift)
                    p = np.polyfit(apg, noise_shifted, 1)
                    noise_shifted1 = np.polyval(p, apg)
                    cal = nrv_cal_s0[v1][valid2]
                    scall = nrv_scall[v1][valid2]
                    nrv_nesz_shifted[v1][valid2] = scall * noise_shifted1 / cal**2
        return nrv_nesz_shifted

    def get_corrected_nesz_vectors(self, polarization, nrv_line, nrv_pixel, nesz):
        """ Load scaling and offset coefficients from files and apply to input  NESZ """
        nrv_nesz_corrected = [np.zeros(p.size)+np.nan for p in nrv_pixel]
        swathBounds = self.import_swathBounds(polarization)
        ns, pb = self.import_denoisingCoefficients(polarization)[:2]
        for swid in self.swath_ids:
            swath_name = f'EW{swid}'
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for v1 in valid1:
                    valid2 = np.where((nrv_pixel[v1] >= frs) * (nrv_pixel[v1] <= lrs))[0]
                    nrv_nesz_corrected[v1][valid2] = nesz[v1][valid2] * ns[swath_name]
                    nrv_nesz_corrected[v1][valid2] += pb[swath_name]
        return nrv_nesz_corrected

    def get_raw_sigma_zero_vectors(self, polarization, nrv_line, nrv_pixel, nrv_cal_s0, average_lines=111):
        """ Read DN_ values from input GeoTIff for a given lines, average in azimuth direction,
        compute sigma0, and return sigma0 for given pixels

        """
        ws2 = np.floor(average_lines / 2)
        raw_sigma_zero = [np.zeros(p.size)+np.nan for p in nrv_pixel]
        src_filename = self.bands()[self.get_band_number(f'DN_{polarization}')]['SourceFilename']
        ds = gdal.Open(src_filename)
        for i in range(nrv_line.shape[0]):
            yoff = max(0, nrv_line[i]-ws2)
            ysize = min(ds.RasterYSize, yoff+ws2)
            line_data = ds.ReadAsArray(yoff=yoff, ysize=average_lines)
            if line_data is None:
                dn_mean = np.zeros(self.shape()[1]) + np.nan
            else:
                dn_mean = line_data.mean(axis=0)
            raw_sigma_zero[i] = dn_mean[nrv_pixel[i]]**2 / nrv_cal_s0[i]**2
        return raw_sigma_zero

    def compute_rqm(self, s0, polarization, nrv_line, nrv_pixel, num_px=100, **kwargs):
        """ Compute Range Quality Metric from the input sigma0 """
        swathBounds = self.import_swathBounds(polarization)
        q_all = {}
        for swid in self.swath_ids[:-1]:
            q_subswath = []
            swath_name = f'EW{swid}'
            swathBound = swathBounds[swath_name]
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                valid1 = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for v1 in valid1:
                    valid2a = np.where((nrv_pixel[v1] >= lrs-num_px) * (nrv_pixel[v1] <= lrs))[0]
                    valid2b = np.where((nrv_pixel[v1] >= lrs+1) * (nrv_pixel[v1] <= lrs+num_px+1))[0]
                    s0a = s0[v1][valid2a]
                    s0b = s0[v1][valid2b]
                    q = np.abs(np.nanmean(s0a) - np.nanmean(s0b)) / (np.nanstd(s0a) + np.nanstd(s0b))
                    q_subswath.append(q)
            q_all[swath_name] = np.array(q_subswath)
        return q_all

    def get_range_quality_metric(self, polarization='HV', **kwargs):
        """ Compute sigma0 with three methods (ESA, SHIFTED, NERSC), compute RQM for each sigma0 """
        nrv_line, nrv_pixel, nrv_noise = self.get_noise_range_vectors(polarization)
        nrv_cal_s0 = self.get_calibration_vectors(polarization, nrv_line, nrv_pixel)
        nrv_scall = self.get_noise_azimuth_vectors(polarization, nrv_line, nrv_pixel)
        nrv_nesz = self.get_raw_nesz_vectors(nrv_noise, nrv_cal_s0, nrv_scall)
        nrv_nesz_shifted = self.get_shifted_nesz_vectors(polarization, nrv_line, nrv_pixel, nrv_noise, nrv_cal_s0, nrv_scall)
        nrv_nesz_corrected = self.get_corrected_nesz_vectors(polarization, nrv_line, nrv_pixel, nrv_nesz_shifted)
        nrv_sigma_zero = self.get_raw_sigma_zero_vectors(polarization, nrv_line, nrv_pixel, nrv_cal_s0)
        s0_esa   = [s0 - n0 for (s0,n0) in zip(nrv_sigma_zero, nrv_nesz)]
        s0_shift = [s0 - n0 for (s0,n0) in zip(nrv_sigma_zero, nrv_nesz_shifted)]
        s0_nersc = [s0 - n0 for (s0,n0) in zip(nrv_sigma_zero, nrv_nesz_corrected)]
        q = [self.compute_rqm(s0, polarization, nrv_line, nrv_pixel, **kwargs) for s0 in [s0_esa, s0_shift, s0_nersc]]

        q_all = {}
        for swid in self.swath_ids[:-1]:
            swath_name = f'EW{swid}'
            q_all[f'RQM_{swath_name}_ESA'] = q[0][swath_name]
            q_all[f'RQM_{swath_name}_SHIFT'] = q[1][swath_name]
            q_all[f'RQM_{swath_name}_NERSC'] = q[2][swath_name]
        return q_all

    def experiment_noiseScaling(self, polarization, min_corr_coef=0.6):
        """ Compute noise scaling coefficients for each range noise line and save as NPZ """
        crop = {'IW':400, 'EW':200}[self.obsMode]
        swathBounds = self.import_swathBounds(polarization)
        nrv_line, nrv_pixel, nrv_noise = self.get_noise_range_vectors(polarization)
        nrv_cal_s0 = self.get_calibration_vectors(polarization, nrv_line, nrv_pixel)
        nrv_scall = self.get_noise_azimuth_vectors(polarization, nrv_line, nrv_pixel)
        nrv_nesz = self.get_raw_nesz_vectors(nrv_noise, nrv_cal_s0, nrv_scall)
        nrv_sigma_zero = self.get_raw_sigma_zero_vectors(polarization, nrv_line, nrv_pixel, nrv_cal_s0)

        results = {}
        results['IPFversion'] = self.IPFversion
        for swid in self.swath_ids:
            swath_name = f'EW{swid}'
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
                valid1 = np.where((nrv_line >= fal) * (nrv_line <= lal))[0]
                for v1 in valid1:
                    valid2 = np.where((nrv_pixel[v1] >= frs+crop) * (nrv_pixel[v1] <= lrs-crop))[0]
                    meanS0 = nrv_sigma_zero[v1][valid2]
                    meanN0 = nrv_nesz[v1][valid2]
                    pixelIndex = nrv_pixel[v1][valid2]
                    (scalingFactor,
                     correlationCoefficient,
                     fitResidual) = fit_noise_scaling_coeff(meanS0, meanN0, pixelIndex)
                    if correlationCoefficient > min_corr_coef:
                        results[swath_name]['sigma0'].append(meanS0)
                        results[swath_name]['noiseEquivalentSigma0'].append(meanN0)
                        results[swath_name]['scalingFactor'].append(scalingFactor)
                        results[swath_name]['correlationCoefficient'].append(correlationCoefficient)
                        results[swath_name]['fitResidual'].append(fitResidual)
        np.savez(self.name.split('.')[0] + '_noiseScaling_new.npz', **results)
