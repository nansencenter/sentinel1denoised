import numpy as np
from osgeo import gdal
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

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
        line = np.array(noiseRangeVector['line'])
        pixel = []
        noise = []

        for pix, n in zip(noiseRangeVector['pixel'], noiseRangeVector['noiseRangeLut']):
            n = np.array(n)
            n[n == 0] = np.nan
            noise.append(n)
            pixel.append(np.array(pix))

        return line, pixel, noise

    def get_calibration_vectors(self, polarization, line, pixel):
        """ Interpolate sigma0 calibration from XML file to the input line/pixel coordinates """
        cal_s0 = [np.zeros(p.size)+np.nan for p in pixel]
        calibrationVector = self.import_calibrationVector(polarization)
        cal_s0arr = np.array(calibrationVector['sigmaNought'])
        swathBounds = self.import_swathBounds(polarization)
        for iSW in self.swath_ids:
            swathBound = swathBounds['%s%s' % (self.obsMode, iSW)]
            line_gpi = ((line >= min(swathBound['firstAzimuthLine'])) *
                        (line <= max(swathBound['lastAzimuthLine'])))
            cal_line = np.array(calibrationVector['line'])
            cal_line_gpi = ((cal_line >= min(swathBound['firstAzimuthLine'])) *
                            (cal_line <= max(swathBound['lastAzimuthLine'])))
            cal_line = cal_line[cal_line_gpi]
            cal_pixel = np.array(calibrationVector['pixel'])[cal_line_gpi][0]
            cal_pixel_gpi = ((cal_pixel >= min(swathBound['firstRangeSample'])) *
                             (cal_pixel <= max(swathBound['lastRangeSample'])))
            cal_pixel = cal_pixel[cal_pixel_gpi]
            cal_s0_pixlin = cal_s0arr[cal_line_gpi][:, cal_pixel_gpi]
            cal_s0_interp = RectBivariateSpline(cal_line, cal_pixel, cal_s0_pixlin, kx=1, ky=1)
            zipped = zip(
                swathBound['firstAzimuthLine'],
                swathBound['lastAzimuthLine'],
                swathBound['firstRangeSample'],
                swathBound['lastRangeSample'],
            )
            for fal, lal, frs, lrs in zipped:
                line_gpi = np.where((line >= fal) * (line <= lal))[0]
                for line_i in line_gpi:
                    pixel_gpi = np.where((pixel[line_i] >= frs) * (pixel[line_i] <= lrs))[0]
                    cal_s0[line_i][pixel_gpi] = cal_s0_interp(
                        line[line_i],
                        pixel[line_i][pixel_gpi])
        return cal_s0

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
        nesz = []
        for n, cal, scall0 in zip(noise, cal_s0, scall):
            n_calib = scall0 * n / cal**2
            nesz.append(n_calib)
        return nesz

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

    def get_shifted_noise_vectors(self, polarization, line, pixel, noise):
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
                    valid2 = np.where(
                        (pixel[v1] >= frs) *
                        (pixel[v1] <= lrs) *
                        (np.isfinite(noise[v1])))[0]
                    valid_pix = pixel[v1][valid2]

                    ba = ba_interpolator(valid_lin, valid_pix).flatten()
                    eap = eap_interpolator(ba).flatten()
                    rsp = rsp_interpolator(valid_lin, valid_pix).flatten()
                    apg = (1/eap/rsp)**2

                    noise_valid = np.array(noise[v1][valid2])
                    noise_interpolator = InterpolatedUnivariateSpline(valid_pix, noise_valid)
                    skip = 4
                    pixel_shift = minimize(cost, 0, args=(valid_pix[skip:-skip], noise_interpolator, apg[skip:-skip])).x[0]

                    noise_shifted0 = noise_interpolator(valid_pix + pixel_shift)
                    p = np.polyfit(apg, noise_shifted0, 1)
                    noise_shifted1 = np.polyval(p, apg)
                    noise_shifted[v1][valid2] = noise_shifted1
        return noise_shifted

    def get_corrected_nesz_vectors(self, polarization, line, pixel, nesz, add_pb=True):
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

    def get_raw_sigma_zero_vectors(self, polarization, line, pixel, cal_s0, average_lines=111):
        """ Read DN_ values from input GeoTIff for a given lines, average in azimuth direction,
        compute sigma0, and return sigma0 for given pixels

        """
        ws2 = np.floor(average_lines / 2)
        raw_sigma_zero = [np.zeros(p.size)+np.nan for p in pixel]
        src_filename = self.bands()[self.get_band_number(f'DN_{polarization}')]['SourceFilename']
        ds = gdal.Open(src_filename)
        for i in range(line.shape[0]):
            yoff = max(0, line[i]-ws2)
            ysize = min(ds.RasterYSize, yoff+ws2)
            line_data = ds.ReadAsArray(yoff=yoff, ysize=average_lines)
            if line_data is None:
                dn_mean = np.zeros(self.shape()[1]) + np.nan
            else:
                dn_mean = line_data.mean(axis=0)
            raw_sigma_zero[i] = dn_mean[pixel[i]]**2 / cal_s0[i]**2
        return raw_sigma_zero

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
                    q = np.abs(np.nanmean(s0a) - np.nanmean(s0b)) / (np.nanstd(s0a) + np.nanstd(s0b))
                    q_subswath.append(q)
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
        nesz_corrected = self.get_corrected_nesz_vectors(polarization, line, pixel, nesz_shifted)
        sigma_zero = self.get_raw_sigma_zero_vectors(polarization, line, pixel, cal_s0)
        s0_esa   = [s0 - n0 for (s0,n0) in zip(sigma_zero, nesz)]
        s0_shift = [s0 - n0 for (s0,n0) in zip(sigma_zero, nesz_shifted)]
        s0_nersc = [s0 - n0 for (s0,n0) in zip(sigma_zero, nesz_corrected)]
        q = [self.compute_rqm(s0, polarization, line, pixel, **kwargs) for s0 in [s0_esa, s0_shift, s0_nersc]]

        q_all = {}
        for swid in self.swath_ids[:-1]:
            swath_name = f'{self.obsMode}{swid}'
            q_all[f'RQM_{swath_name}_ESA'] = q[0][swath_name]
            q_all[f'RQM_{swath_name}_SHIFT'] = q[1][swath_name]
            q_all[f'RQM_{swath_name}_NERSC'] = q[2][swath_name]
        return q_all

    def experiment_noiseScaling(self, polarization, average_lines=777, zoom_step=2):
        """ Compute noise scaling coefficients for each range noise line and save as NPZ """
        crop = {'IW':400, 'EW':200}[self.obsMode]
        swathBounds = self.import_swathBounds(polarization)
        line, pixelrv0, noise0 = self.get_noise_range_vectors(polarization)
        # zoom:
        pixel = [np.arange(p[0], p[-1], zoom_step) for p in pixelrv0]
        noise = [interp1d(p, n)(p2) for (p,n,p2) in zip(pixelrv0, noise0, pixel)]
        noise_shifted = self.get_shifted_noise_vectors(polarization, line, pixel, noise)
        cal_s0 = self.get_calibration_vectors(polarization, line, pixel)
        scall = self.get_noise_azimuth_vectors(polarization, line, pixel)
        nesz = self.calibrate_noise_vectors(noise_shifted, cal_s0, scall)

        sigma_zero = self.get_raw_sigma_zero_vectors(
            polarization, line, pixel, cal_s0, average_lines=average_lines)

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
                    meanS0 = sigma_zero[v1][valid2]
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
        np.savez(self.name.split('.')[0] + '_noiseScaling_new.npz', **results)

    def experiment_powerBalancing(self, polarization, average_lines=777, zoom_step=2):
        """ Compute power balancing coefficients for each range noise line and save as NPZ """
        crop = {'IW':400, 'EW':200}[self.obsMode]
        swathBounds = self.import_swathBounds(polarization)
        line, pixelrv0, noise0 = self.get_noise_range_vectors(polarization)
        # zoom:
        pixel = [np.arange(p[0], p[-1], zoom_step) for p in pixelrv0]
        noise = [interp1d(p, n)(p2) for (p,n,p2) in zip(pixelrv0, noise0, pixel)]
        noise_shifted = self.get_shifted_noise_vectors(polarization, line, pixel, noise)
        cal_s0 = self.get_calibration_vectors(polarization, line, pixel)
        scall = self.get_noise_azimuth_vectors(polarization, line, pixel)
        nesz = self.calibrate_noise_vectors(noise_shifted, cal_s0, scall)
        nesz_corrected = self.get_corrected_nesz_vectors(polarization, line, pixel, nesz, add_pb=False)
        sigma_zero = self.get_raw_sigma_zero_vectors(
            polarization, line, pixel, cal_s0, average_lines=average_lines)

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
                meanS0 = sigma_zero[li][valid2]
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

        np.savez(self.name.split('.')[0] + '_powerBalancing_new.npz', **results)
