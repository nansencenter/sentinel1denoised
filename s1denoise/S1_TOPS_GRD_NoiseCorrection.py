# Sentinel1Denoised performs various corrections on sentinel1 images
# Copyright (C) 2016-2018 Nansen Environmental and Remote Sensing Center,
#                         Jeong-Won Park, Anton Korosov

# References
# R1: Efficient Thermal Noise Removal for Sentinel-1 TOPSAR Cross-Polarization Channel
#     Park et al., 2018, IEEE TGRS 56(3):1555-1565. doi:10.1109/TGRS.2017.2765248
# R2: Sentinel-1 Level 1 Detailed Algorithm Definition. Available from
# https://sentinel.esa.int/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition
# R3: Digital Processing of Synthetic Aperture Radar Data: Algorithms and Implementation
#     Cumming and Wong, 2005, Artech House
# R4: Textural Noise Correction for Sentinel-1 TOPSAR Cross-Polarization Channel Images
#     Park et al., 2019, IEEE TGRS 57(6):4040-4049. doi:10.1109/TGRS.2018.2889381


import os, sys, glob, warnings, zipfile, requests, subprocess
import numpy as np
from datetime import datetime, timedelta
from xml.dom.minidom import parse, parseString
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.ndimage import convolve, uniform_filter
from scipy.optimize import fminbound

from nansat import Nansat

warnings.simplefilter("ignore")


# define radar related constants
SPEED_OF_LIGHT = 299792458.
RADAR_FREQUENCY = 5.405000454334350e+09
RADAR_WAVELENGTH = SPEED_OF_LIGHT / RADAR_FREQUENCY
ANTENNA_STEERING_RATE = { 'IW1': 1.590368784,
                          'IW2': 0.979863325,
                          'IW3': 1.397440818,
                          'EW1': 2.390895448,
                          'EW2': 2.811502724,
                          'EW3': 2.366195855,
                          'EW4': 2.512694636,
                          'EW5': 2.122855427 }    # degrees per second. Available from AUX_INS

def get_DOM_subElement(element, tags):
    ''' Get sub-element from XML DOM element based on tags '''
    for tag in tags:
        element = element.getElementsByTagName(tag)[0]
    return element

def get_DOM_nodeValue(element, tags, oType='str'):
    ''' Get value of XML DOM sub-element based on tags '''
    if oType not in ['str', 'int', 'float']:
        raise ValueError('see error message.')
    value = get_DOM_subElement(element, tags).childNodes[0].nodeValue.split()
    if oType == 'str':
        value = [str(v) for v in value]
    elif oType == 'int':
        value = [int(round(float(v))) for v in value]
    elif oType == 'float':
        value = [float(v) for v in value]
    if len(value)==1:
        value = value[0]
    return value

def cubic_hermite_interpolation(x,y,xi):
    ''' Get interpolated value for given time '''
    return np.polynomial.hermite.hermval(xi, np.polynomial.hermite.hermfit(x,y,deg=3))


def range_to_target(satPosVec, lookVec, terrainHeight=0):
    ''' Compute slant range distance to target on WGS-84 Earth ellipsoid '''
    # See Step 4 of Appendix 12A in the reference, R3.
    A_e = 6378137.0
    B_e = A_e * (1 - 1./298.257223563)
    A_e += terrainHeight
    B_e += terrainHeight
    epsilon = (A_e**2-B_e**2)/B_e**2
    F = ( (np.dot(satPosVec,lookVec) + epsilon * satPosVec[2] * lookVec[2])
          / (1 + epsilon * lookVec[2]**2) )
    G = ( (np.dot(satPosVec,satPosVec) - A_e**2 + epsilon * satPosVec[2]**2)
          / (1 + epsilon * lookVec[2]**2) )
    return -F - np.sqrt(F**2 - G)

def planar_rotation(rotAxis, inputVec, rotAngle):
    ''' Planar rotation about a given axis '''
    rotAxis = rotAxis / np.linalg.norm(rotAxis)
    sinAng = np.sin(np.deg2rad(rotAngle))
    cosAng = np.cos(np.deg2rad(rotAngle))
    d11 = (1 - cosAng) * rotAxis[0]**2 + cosAng
    d12 = (1 - cosAng) * rotAxis[0] * rotAxis[1] - sinAng * rotAxis[2]
    d13 = (1 - cosAng) * rotAxis[0] * rotAxis[2] + sinAng * rotAxis[1]
    d21 = (1 - cosAng) * rotAxis[0] * rotAxis[1] + sinAng * rotAxis[2]
    d22 = (1 - cosAng) * rotAxis[1]**2 + cosAng
    d23 = (1 - cosAng) * rotAxis[1] * rotAxis[2] - sinAng * rotAxis[0]
    d31 = (1 - cosAng) * rotAxis[0] * rotAxis[2] - sinAng * rotAxis[1]
    d32 = (1 - cosAng) * rotAxis[1] * rotAxis[2] + sinAng * rotAxis[0]
    d33 = (1 - cosAng) * rotAxis[2]**2 + cosAng
    outputVec = np.array([ d11 * inputVec[0] + d12 * inputVec[1] + d13 * inputVec[2],
                           d21 * inputVec[0] + d22 * inputVec[1] + d23 * inputVec[2],
                           d31 * inputVec[0] + d32 * inputVec[1] + d33 * inputVec[2]  ])
    return outputVec

def est_shift(reference, test, oversampling=10):
    ''' Estimate relative shift '''
    # test signal must be longer than reference signal
    lags = np.arange(len(test) - len(reference) + 1)
    cc = np.zeros(len(lags))
    for li, lag in enumerate(lags):
        cc[li] = np.corrcoef(reference, test[lag:lag+len(reference)])[0,1]
    x = np.arange(len(cc))
    xi = np.linspace(0, len(cc), (len(cc)-1) * oversampling +1)
    cci = InterpolatedUnivariateSpline(x, cc)(xi)
    return xi[np.argmax(cci)] - len(cc)//2

def fillGaps(dataArray, filterSize=3):
    ''' Fill gaps using mirrored vector in each dimension '''
    if filterSize%2==0:
        filterSize += 1
    fs = filterSize//2
    nRows, nCols = dataArray.shape
    for li in range(nRows):
        fi = np.where(np.isfinite(dataArray[li,:]))[0]
        if len(fi)<2:
            continue
        fiMin, fiMax = fi.min(), fi.max()
        if fiMax!=(nCols-1):
            dataArray[li,:][fiMax+1:min(nCols,fiMax*2)] = dataArray[li,:][fiMax-1:max(0,fiMax-(nCols-fiMax)):-1]
        if fiMin!=0:
            dataArray[li,:][0:min(fiMin,nCols-fiMin-1)] = dataArray[li,:][min(nCols-1,2*fiMin):fiMin:-1]
    for li in range(nCols):
        fi = np.where(np.isfinite(dataArray[:,li]))[0]
        if len(fi)<2:
            continue
        fiMin, fiMax = fi.min(), fi.max()
        if fiMax!=(nRows-1):
            dataArray[:,li][fiMax+1:min(nRows,fiMax*2)] = dataArray[:,li][fiMax-1:max(0,fiMax-(nRows-fiMax)):-1]
        if fiMin!=0:
            dataArray[:,li][0:min(fiMin,nRows-fiMin-1)] = dataArray[:,li][min(nRows-1,2*fiMin):fiMin:-1]
    ri,ci = np.where(np.isnan(dataArray))
    while len(ri)!=0:
        for r,c in zip(ri,ci):
            dataArray[r,c] = np.nanmean(dataArray[max(0,r-fs):min(nRows,r+fs+1),max(0,c-fs):min(nCols,c+fs+1)])
        ri,ci = np.where(np.isnan(dataArray))
    return dataArray


class Sentinel1Image(Nansat):

    def __init__(self, filename, mapperName='sentinel1_l1', logLevel=30):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__( self, filename,
                         mapperName=mapperName, logLevel=logLevel)
        if ( self.filename.split('/')[-1][4:16]
             not in [ 'IW_GRDH_1SDH',
                      'IW_GRDH_1SDV',
                      'EW_GRDM_1SDH',
                      'EW_GRDM_1SDV'  ] ):
             raise ValueError( 'Source file must be Sentinel-1A/1B '
                 'IW_GRDH_1SDH, IW_GRDH_1SDV, EW_GRDM_1SDH, or EW_GRDM_1SDV product.' )
        self.platform = self.filename.split('/')[-1][:3]    # S1A or S1B
        self.obsMode = self.filename.split('/')[-1][4:6]    # IW or EW
        txPol = self.filename.split('/')[-1][15]    # H or V
        self.annotationXML = {}
        self.calibrationXML = {}
        self.noiseXML = {}
        if zipfile.is_zipfile(self.filename):
            zf = zipfile.PyZipFile(self.filename)
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
            zf.close()
        else:
            annotationFiles = [fn for fn in glob.glob(self.filename+'/annotation/*') if 's1' in fn]
            calibrationFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*')
                                if 'calibration-s1' in fn]
            noiseFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*')
                          if 'noise-s1' in fn]
            for polarization in [txPol + 'H', txPol + 'V']:
                self.annotationXML[polarization] = parseString(
                    [open(fn).read() for fn in annotationFiles if polarization.lower() in fn][0])
                self.calibrationXML[polarization] = parseString(
                    [open(fn).read() for fn in calibrationFiles if polarization.lower() in fn][0])
                self.noiseXML[polarization] = parseString(
                    [open(fn).read() for fn in noiseFiles if polarization.lower() in fn][0])
            self.manifestXML = parseString(
                open(glob.glob(self.filename+'/manifest.safe')[0]).read())
        # scene center time will be used as the reference for relative azimuth time in seconds
        self.time_coverage_center = ( self.time_coverage_start + timedelta(
            seconds=(self.time_coverage_end - self.time_coverage_start).total_seconds()/2) )
        # get processor version of Sentinel-1 IPF (Instrument Processing Facility)
        self.IPFversion = float(self.manifestXML.getElementsByTagName('safe:software')[0]
                                .attributes['version'].value)
        if self.IPFversion < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Noise correction cannot be achieved using this module. '
                  'Denoising vectors in annotation file are not qualified.\n')
            return
        elif 2.43 <= self.IPFversion < 2.53:
            print('\nWARNING: IPF version of input image is lower than 2.53! '
                  'Noise correction result might be wrong.\n')
        # get the auxiliary calibration file
        resourceList = self.manifestXML.getElementsByTagName('resource')
        if resourceList==[]:
            resourceList = self.manifestXML.getElementsByTagName('safe:resource')
        for resource in resourceList:
            if resource.attributes['role'].value=='AUX_CAL':
                auxCalibFilename = resource.attributes['name'].value.split('/')[-1]
        self.set_aux_data_dir()
        self.download_aux_calibration(auxCalibFilename, self.platform.lower())
        self.auxiliaryCalibrationXML = parse(self.auxiliaryCalibration_file)

    def set_aux_data_dir(self):
        """ Set directory where aux calibration data is stored """
        self.aux_data_dir = os.path.join(os.environ.get('XDG_DATA_HOME', os.environ.get('HOME')),
                                         '.s1denoise')
        if not os.path.exists(self.aux_data_dir):
            os.makedirs(self.aux_data_dir)

    def download_aux_calibration(self, filename, platform):
        """ Download auxiliary calibration files form ESA in self.aux_data_dir """
        cal_file = os.path.join(self.aux_data_dir, filename, 'data', '%s-aux-cal.xml' % platform)
        cal_file_tgz = os.path.join(self.aux_data_dir, filename + '.TGZ')
        if not os.path.exists(cal_file):
            parts = filename.split('_')
            cal_url = ('https://qc.sentinel1.eo.esa.int/product/%s/%s_%s/%s/%s.TGZ'
                       % (parts[0], parts[1], parts[2], parts[3][1:], filename))
            r = requests.get(cal_url, stream=True)
            with open(cal_file_tgz, "wb") as f:
                f.write(r.content)
            subprocess.call(['tar', '-xzvf', cal_file_tgz, '-C', self.aux_data_dir])
        self.auxiliaryCalibration_file = cal_file

    def import_antennaPattern(self, polarization):
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
                           for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in antennaPatternList:
            swath = get_DOM_nodeValue(iList,['swath'])
            for k in antennaPattern[swath].keys():
                if k=='azimuthTime':
                    antennaPattern[swath][k].append(
                        datetime.strptime(get_DOM_nodeValue(iList,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                else:
                    antennaPattern[swath][k].append(
                        get_DOM_nodeValue(iList,[k],'float'))
        return antennaPattern

    def import_azimuthAntennaElementPattern(self, polarization):
        ''' Import azimuth antenna element pattern from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        azimuthAntennaElementPattern = { '%s%s' % (self.obsMode, li):
                                             { 'azimuthAngleIncrement':[],
                                               'azimuthAntennaElementPattern':[],
                                               'absoluteCalibrationConstant':[],
                                               'noiseCalibrationFactor':[] }
                                         for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            pol = get_DOM_nodeValue(iList,['polarisation'])
            if (swath in azimuthAntennaElementPattern.keys()) and (pol==polarization):
                elem = iList.getElementsByTagName('azimuthAntennaElementPattern')[0]
                for k in azimuthAntennaElementPattern[swath].keys():
                    if k=='azimuthAngleIncrement':
                        azimuthAntennaElementPattern[swath][k] = (
                            get_DOM_nodeValue(elem,[k],'float') )
                    elif k=='azimuthAntennaElementPattern':
                        azimuthAntennaElementPattern[swath][k] = (
                            get_DOM_nodeValue(elem,['values'],'float') )
                    else:
                        azimuthAntennaElementPattern[swath][k] = (
                            get_DOM_nodeValue(iList,[k],'float') )
        return azimuthAntennaElementPattern

    def import_azimuthFmRate(self, polarization):
        ''' Import azimuth frequency modulation rate from annotation XML DOM '''
        azimuthFmRateList = self.annotationXML[polarization].getElementsByTagName('azimuthFmRate')
        azimuthFmRate = { 'azimuthTime':[],
                          't0':[],
                          'azimuthFmRatePolynomial':[] }
        for iList in azimuthFmRateList:
            for k in azimuthFmRate.keys():
                if k=='azimuthTime':
                    azimuthFmRate[k].append(
                        datetime.strptime(get_DOM_nodeValue(iList,[k]),'%Y-%m-%dT%H:%M:%S.%f'))
                else:
                    azimuthFmRate[k].append(
                        get_DOM_nodeValue(iList,[k],'float'))
        return azimuthFmRate

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

    def import_denoisingCoefficients(self, polarization):
        ''' Import denoising coefficients '''
        satID = self.filename.split('/')[-1][:3]
        denoParams = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'denoising_parameters_%s.npz' % satID),
                             allow_pickle=True)[polarization].item()
        noiseScalingParameters = {}
        powerBalancingParameters = {}
        extraScalingParameters = {}
        extraScalingParameters['SNR'] = []
        noiseVarianceParameters = {}
        IPFversion = float(self.IPFversion)
        sensingDate = datetime.strptime(self.filename.split('/')[-1].split('_')[4], '%Y%m%dT%H%M%S')
        if satID=='S1B' and IPFversion==2.72 and sensingDate >= datetime(2017,1,16,13,42,34):
            # Adaption for special case.
            # ESA abrubtly changed scaling LUT in AUX_PP1 from 20170116 while keeping the IPFv.
            # After this change, the scaling parameters seems be much closer to those of IPFv 2.8.
            IPFversion = 2.8
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            if 'noiseScalingParameters' in denoParams.keys():
                try:
                    noiseScalingParameters[subswathID] = (
                        denoParams['noiseScalingParameters'][subswathID]['%.1f' % IPFversion])
                except:
                    print('WARNING: noise scaling parameters for subswath %s (IPF:%s) is missing.'
                          % (subswathID, self.IPFversion))
                    noiseScalingParameters[subswathID] = 1.0
            else:
                print('WARNING: noiseScalingParameters field is missing.')
                noiseScalingParameters[subswathID] = 1.0
            if 'powerBalancingParameters' in denoParams.keys():
                try:
                    powerBalancingParameters[subswathID] = (
                        denoParams['powerBalancingParameters'][subswathID]['%.1f' % IPFversion])
                except:
                    print('WARNING: power balancing parameters for subswath %s (IPF:%s) is missing.'
                          % (subswathID, self.IPFversion))
                    powerBalancingParameters[subswathID] = 0.0
            else:
                print('WARNING: powerBalancingParameters field is missing.')
                powerBalancingParameters[subswathID] = 0.0
            if 'extraScalingParameters' in denoParams.keys():
                try:
                    extraScalingParameters[subswathID] = (
                        denoParams['extraScalingParameters'][subswathID])
                    extraScalingParameters['SNR'] = (
                        denoParams['extraScalingParameters']['SNNR'])
                except:
                    print('WARNING: extra scaling parameters for subswath %s (IPF:%s) is missing.'
                          % (subswathID, self.IPFversion))
                    extraScalingParameters['SNNR'] = np.linspace(-30,+30,601)
                    extraScalingParameters[subswathID] = np.ones(601)
            else:
                print('WARNING: extraScalingParameters field is missing.')
                extraScalingParameters['SNNR'] = np.linspace(-30,+30,601)
                extraScalingParameters[subswathID] = np.ones(601)
            if 'noiseVarianceParameters' in denoParams.keys():
                try:
                    noiseVarianceParameters[subswathID] = (
                        denoParams['noiseVarianceParameters'][subswathID] )
                except:
                    print('WARNING: noise variance parameters for subswath %s (IPF:%s) is missing.'
                          % (subswathID, self.IPFversion))
                    noiseVarianceParameters[subswathID] = 0.0
            else:
                print('WARNING: noiseVarianceParameters field is missing.')
                noiseVarianceParameters[subswathID] = 0.0
        return ( noiseScalingParameters, powerBalancingParameters, extraScalingParameters,
                 noiseVarianceParameters )

    def import_elevationAntennaPattern(self, polarization):
        ''' Import elevation antenna pattern from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        elevationAntennaPattern = { '%s%s' % (self.obsMode, li):
                                        { 'elevationAngleIncrement':[],
                                          'elevationAntennaPattern':[],
                                          'absoluteCalibrationConstant':[],
                                          'noiseCalibrationFactor':[] }
                                    for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            pol = get_DOM_nodeValue(iList,['polarisation'])
            if (swath in elevationAntennaPattern.keys()) and (pol==polarization):
                elem = iList.getElementsByTagName('elevationAntennaPattern')[0]
                for k in elevationAntennaPattern[swath].keys():
                    if k=='elevationAngleIncrement':
                        elevationAntennaPattern[swath][k] = (
                            get_DOM_nodeValue(elem,[k],'float') )
                    elif k=='elevationAntennaPattern':
                        elevationAntennaPattern[swath][k] = (
                            get_DOM_nodeValue(elem,['values'],'float') )
                    else:
                        elevationAntennaPattern[swath][k] = (
                            get_DOM_nodeValue(iList,[k],'float') )
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
                               for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
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
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
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

    def import_orbit(self, polarization):
        ''' Import orbit information from annotation XML DOM '''
        orbitList = self.annotationXML[polarization].getElementsByTagName('orbit')
        orbit = { 'time':[],
                  'position':{'x':[], 'y':[], 'z':[]},
                  'velocity':{'x':[], 'y':[], 'z':[]} }
        for iList in orbitList:
            orbit['time'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['time']), '%Y-%m-%dT%H:%M:%S.%f'))
            for k in orbit['position'].keys():
                orbit['position'][k].append(
                    get_DOM_nodeValue(iList,['position',k],'float'))
            for k in orbit['velocity'].keys():
                orbit['velocity'][k].append(
                    get_DOM_nodeValue(iList,['velocity',k],'float'))
        return orbit

    def import_processorScalingFactor(self, polarization):
        ''' Import swath processing scaling factors from annotation XML DOM '''
        swathProcParamsList = self.annotationXML[polarization].getElementsByTagName(
            'swathProcParams')
        processorScalingFactor = {}
        for iList in swathProcParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            processorScalingFactor[swath] = get_DOM_nodeValue(
                iList,['processorScalingFactor'],'float')
        return processorScalingFactor

    def import_swathBounds(self, polarization):
        ''' Import swath bounds information from annotation XML DOM '''
        swathMergeList = self.annotationXML[polarization].getElementsByTagName('swathMerge')
        swathBounds = { '%s%s' % (self.obsMode, li):
                            { 'azimuthTime':[],
                              'firstAzimuthLine':[],
                              'firstRangeSample':[],
                              'lastAzimuthLine':[],
                              'lastRangeSample':[] }
                        for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
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

    def azimuthFmRateAtGivenTime(self, polarization, relativeAzimuthTime, slantRangeTime):
        ''' Get azimuth frequency modulation rate for given time vectors

        Returns
        -------
        vector for all pixels in azimuth direction
        '''
        if relativeAzimuthTime.size != slantRangeTime.size:
            raise ValueError('relativeAzimuthTime and slantRangeTime must have the same dimension')
        azimuthFmRate = self.import_azimuthFmRate(polarization)
        azimuthFmRatePolynomial = np.array(azimuthFmRate['azimuthFmRatePolynomial'])
        t0 = np.array(azimuthFmRate['t0'])
        xp = np.array([ (t-self.time_coverage_center).total_seconds()
                        for t in azimuthFmRate['azimuthTime'] ])
        azimuthFmRateAtGivenTime = []
        for tt in zip(relativeAzimuthTime,slantRangeTime):
            fp = (   azimuthFmRatePolynomial[:,0]
                   + azimuthFmRatePolynomial[:,1] * (tt[1]-t0)**1
                   + azimuthFmRatePolynomial[:,2] * (tt[1]-t0)**2 )
            azimuthFmRateAtGivenTime.append(np.interp(tt[0], xp, fp))
        return np.squeeze(azimuthFmRateAtGivenTime)

    def focusedBurstLengthInTime(self, polarization):
        ''' Get focused burst length in zero-Doppler time domain

        Returns
        -------
        focusedBurstLengthInTime : dict
            one values for each subswath (different for IW and EW)
        '''
        azimuthFrequency = get_DOM_nodeValue(
            self.annotationXML[polarization],['azimuthFrequency'],'float')
        azimuthTimeIntevalInSLC = 1. / azimuthFrequency
        inputDimensionsList = self.annotationXML[polarization].getElementsByTagName(
            'inputDimensions')
        focusedBurstLengthInTime = {}
        # nominalLinesPerBurst should be smaller than the real values
        nominalLinesPerBurst = {'IW':1450, 'EW':1100}[self.obsMode]
        for iList in inputDimensionsList:
            swath = get_DOM_nodeValue(iList,['swath'],'str')
            numberOfInputLines = get_DOM_nodeValue(iList,['numberOfInputLines'],'int')
            numberOfBursts = max(
                [ primeNumber for primeNumber in range(1,numberOfInputLines//nominalLinesPerBurst+1)
                  if (numberOfInputLines % primeNumber)==0 ] )
            if (numberOfInputLines % numberOfBursts)==0:
                focusedBurstLengthInTime[swath] = (
                    numberOfInputLines / numberOfBursts * azimuthTimeIntevalInSLC )
            else:
                raise ValueError('number of bursts cannot be determined.')
        return focusedBurstLengthInTime

    def geolocationGridPointInterpolator(self, polarization, itemName):
        ''' Generate interpolator for items in geolocation grid point list '''
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        if itemName not in geolocationGridPoint.keys():
            raise ValueError('%s is not in the geolocationGridPoint list.' % itemName)
        x = np.unique(geolocationGridPoint['pixel'])
        y = np.unique(geolocationGridPoint['line'])
        if itemName=='azimuthTime':
            z = [ (t-self.time_coverage_center).total_seconds()
                  for t in geolocationGridPoint['azimuthTime'] ]
            z = np.reshape(z,(len(y),len(x)))
        else:
            z = np.reshape(geolocationGridPoint[itemName],(len(y),len(x)))
        interpolator = RectBivariateSpline(y, x, z)
        return interpolator

    def orbitAtGivenTime(self, polarization, relativeAzimuthTime):
        ''' Interpolate orbit parameters for given time vector '''
        stateVectors = self.import_orbit(polarization)
        orbitTime = np.array([ (t-self.time_coverage_center).total_seconds()
                                for t in stateVectors['time'] ])
        orbitAtGivenTime = { 'relativeAzimuthTime':relativeAzimuthTime,
                             'positionXYZ':[],
                             'velocityXYZ':[] }
        for t in relativeAzimuthTime:
            useIndices = sorted(np.argsort(abs(orbitTime-t))[:4])
            for k in ['position', 'velocity']:
                orbitAtGivenTime[k+'XYZ'].append([
                    cubic_hermite_interpolation(orbitTime[useIndices],
                        np.array(stateVectors[k][component])[useIndices], t)
                    for component in ['x','y','z'] ])
        for k in ['positionXYZ', 'velocityXYZ']:
            orbitAtGivenTime[k] = np.squeeze(orbitAtGivenTime[k])
        return orbitAtGivenTime

    def subswathCenterSampleIndex(self, polarization):
        ''' Range center pixel indices along azimuth for each subswath '''
        swathBounds = self.import_swathBounds(polarization)
        subswathCenterSampleIndex = {}
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfLines = (   np.array(swathBounds[subswathID]['lastAzimuthLine'])
                              - np.array(swathBounds[subswathID]['firstAzimuthLine']) + 1 )
            midPixelIndices = (   np.array(swathBounds[subswathID]['firstRangeSample'])
                                + np.array(swathBounds[subswathID]['lastRangeSample']) ) / 2.
            subswathCenterSampleIndex[subswathID] = int(round(
                np.sum(midPixelIndices * numberOfLines) / np.sum(numberOfLines) ))
        return subswathCenterSampleIndex

    def calibrationVectorMap(self, polarization):
        ''' Convert calibration vectors into full grid pixels '''
        calibrationVector = self.import_calibrationVector(polarization)
        swathBounds = self.import_swathBounds(polarization)
        subswathIndexMap = self.subswathIndexMap(polarization)
        calibrationVectorMap = np.ones(self.shape()) * np.nan
        # subswath-wise processing is required
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            swathBound = swathBounds['%s%s' % (self.obsMode, iSW)]
            line = np.array(calibrationVector['line'])
            valid = (   (line >= min(swathBound['firstAzimuthLine']))
                      * (line <= max(swathBound['lastAzimuthLine'])) )
            line = line[valid]
            pixel = np.array(calibrationVector['pixel'])[valid]
            sigmaNought = np.array(calibrationVector['sigmaNought'])[valid]
            xBins = np.arange(min(swathBound['firstRangeSample']),
                              max(swathBound['lastRangeSample']) + 1)
            zi = np.ones((valid.sum(), len(xBins))) * np.nan
            for li,y in enumerate(line):
                x = np.array(pixel[li])
                z = np.array(sigmaNought[li])
                valid = (subswathIndexMap[y,x]==iSW) * (z > 0)
                if valid.sum()==0:
                    continue
                zi[li,:] = InterpolatedUnivariateSpline(x[valid], z[valid])(xBins)
            valid = np.isfinite(np.sum(zi,axis=1))
            interpFunc = RectBivariateSpline(xBins, line[valid], zi[valid,:].T, kx=1, ky=1)
            valid = (subswathIndexMap==iSW)
            calibrationVectorMap[valid] = interpFunc(np.arange(self.shape()[1]),
                                                     np.arange(self.shape()[0])).T[valid]
        return calibrationVectorMap

    def refinedElevationAngleInterpolator(self, polarization):
        ''' Generate elevation angle interpolator using the refined elevation angle
            calculated from orbit vectors and WGS-84 ellipsoid '''
        angleStep = 1e-3
        maxIter = 100
        distanceThreshold = 1e-2
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        line = geolocationGridPoint['line']
        uniqueLine = np.unique(line)
        pixel = geolocationGridPoint['pixel']
        uniquePixel = np.unique(pixel)
        azimuthTimeIntp = self.geolocationGridPointInterpolator(polarization, 'azimuthTime')
        slantRangeTimeIntp = self.geolocationGridPointInterpolator(polarization, 'slantRangeTime')
        subswathIndexMap = self.subswathIndexMap(polarization)
        orbits = self.orbitAtGivenTime(polarization,
            azimuthTimeIntp(uniqueLine, uniquePixel).reshape(len(uniqueLine) * len(uniquePixel)) )
        elevationAngle = []
        for li in range(len(uniqueLine) * len(uniquePixel)):
            positionVector = orbits['positionXYZ'][li]
            velocityVector = orbits['velocityXYZ'][li]
            slantRangeDistance = (SPEED_OF_LIGHT / 2 * slantRangeTimeIntp(line[li], pixel[li])).item()
            lookVector = np.cross(velocityVector, positionVector)
            lookVector /= np.linalg.norm(lookVector)
            rotationAxis = np.cross(positionVector, lookVector)
            rotationAxis /= np.linalg.norm(rotationAxis)
            depressionAngle = 45.
            nIter = 0
            status = False
            while not (status or (nIter >= maxIter)):
                rotatedLookVector1 = planar_rotation(
                    rotationAxis, lookVector, depressionAngle)
                rotatedLookVector2 = planar_rotation(
                    rotationAxis, lookVector, depressionAngle + angleStep)
                err1 = range_to_target(positionVector, rotatedLookVector1) - slantRangeDistance
                err2 = range_to_target(positionVector, rotatedLookVector2) - slantRangeDistance
                depressionAngle += ( err1 / ((err1 - err2) / angleStep) )
                status = np.abs(err1).max() < distanceThreshold
                nIter += 1
            elevationAngle.append(90. - depressionAngle)
        elevationAngle = np.reshape(elevationAngle, (len(uniqueLine), len(uniquePixel)))
        interpolator = RectBivariateSpline(uniqueLine, uniquePixel, elevationAngle)
        return interpolator

    def rollAngleInterpolator(self, polarization, source='annotated'):
        ''' Generate roll angle interpolator '''
        antennaPattern = self.import_antennaPattern(polarization)
        relativeAzimuthTime = []
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            relativeAzimuthTime.append([ (t-self.time_coverage_center).total_seconds()
                                         for t in antennaPattern[subswathID]['azimuthTime'] ])
        relativeAzimuthTime = np.hstack(relativeAzimuthTime)
        sortIndex = np.argsort(relativeAzimuthTime)
        if source=='annotated':
            rollAngle = []
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                rollAngle.append(antennaPattern[subswathID]['roll'])
            relativeAzimuthTime = np.hstack(relativeAzimuthTime)
            rollAngle = np.hstack(rollAngle)
        elif source=='nominal':
            positionXYZ = self.orbitAtGivenTime(polarization, relativeAzimuthTime)['positionXYZ']
            satelliteLatitude = np.arctan2(positionXYZ[:,2],
                                           np.sqrt(positionXYZ[:,0]**2 + positionXYZ[:,1]**2))
            r_major = 6378137.0            # WGS84 semi-major axis
            r_minor = 6356752.314245179    # WGS84 semi-minor axis
            earthRadius = np.sqrt(  (  (r_major**2 * np.cos(satelliteLatitude))**2
                                     + (r_minor**2 * np.sin(satelliteLatitude))**2)
                                  / (  (r_major * np.cos(satelliteLatitude))**2
                                     + (r_minor * np.sin(satelliteLatitude))**2) )
            satelliteAltitude = np.linalg.norm(positionXYZ, axis=1) - earthRadius
            # see Eq.9-19 in the reference R2.
            rollAngle = 29.45 - 0.0566*(satelliteAltitude/1000. - 711.7)
        rollAngleIntp = InterpolatedUnivariateSpline(
            relativeAzimuthTime[sortIndex], rollAngle[sortIndex])
        azimuthTimeIntp = self.geolocationGridPointInterpolator(polarization, 'azimuthTime')
        geolocationGridPoint = self.import_geolocationGridPoint(polarization)
        line = geolocationGridPoint['line']
        uniqueLine = np.unique(line)
        pixel = geolocationGridPoint['pixel']
        uniquePixel = np.unique(pixel)
        interpolator = RectBivariateSpline(
            uniqueLine, uniquePixel, rollAngleIntp(azimuthTimeIntp(uniqueLine, uniquePixel)))
        return interpolator

    def elevationAntennaPatternInterpolator(self, polarization):
        ''' Generate elevation antenna pattern interpolator

        Inerpolator that retuns antenna pattern for given elevation angle
        '''
        #elevationAngleIntp = self.geolocationGridPointInterpolator(polarization, 'elevationAngle')
        elevationAngleIntp = self.refinedElevationAngleInterpolator(polarization)
        elevationAngleMap = np.squeeze(elevationAngleIntp(
            np.arange(self.shape()[0]), np.arange(self.shape()[1])))
        rollAngleIntp = self.rollAngleInterpolator(polarization)
        rollAngleMap = np.squeeze(rollAngleIntp(
            np.arange(self.shape()[0]), np.arange(self.shape()[1])))
        boresightAngleMap = elevationAngleMap - rollAngleMap
        del elevationAngleMap, rollAngleMap
        subswathIndexMap = self.subswathIndexMap(polarization)
        elevationAntennaPatternMap = np.ones(self.shape())
        elevationAntennaPatternLUT = self.import_elevationAntennaPattern(polarization)
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            recordLength = len(elevationAntennaPatternLUT[subswathID]['elevationAntennaPattern'])/2
            angleLUT = ( np.arange(-(recordLength//2),+(recordLength//2)+1)
                         * elevationAntennaPatternLUT[subswathID]['elevationAngleIncrement'] )
            amplitudeLUT = np.array(
                elevationAntennaPatternLUT[subswathID]['elevationAntennaPattern'])
            amplitudeLUT = np.sqrt(amplitudeLUT[0::2]**2 + amplitudeLUT[1::2]**2)
            interpolator = InterpolatedUnivariateSpline(angleLUT, np.sqrt(amplitudeLUT))
            valid = (subswathIndexMap==iSW)
            elevationAntennaPatternMap[valid] = interpolator(boresightAngleMap)[valid]
        interpolator = RectBivariateSpline(
            np.arange(self.shape()[0]), np.arange(self.shape()[1]), elevationAntennaPatternMap)
        return interpolator

    def rangeSpreadingLossInterpolator(self, polarization):
        ''' Generate range spreading loss interpolator

        Interpolator that calculate range spreading loss for a given elevation angle'''
        # See Section 9.6 of the reference, R2.
        referenceRange = float(self.annotationXML[polarization].getElementsByTagName(
            'referenceRange')[0].childNodes[0].nodeValue)
        slantRangeTimeIntp = self.geolocationGridPointInterpolator(polarization, 'slantRangeTime')
        slantRangeTimeMap = np.squeeze(slantRangeTimeIntp(
            np.arange(self.shape()[0]), np.arange(self.shape()[1])))
        rangeSpreadingLoss = (referenceRange / slantRangeTimeMap / SPEED_OF_LIGHT * 2)**(3./2.)
        interpolator = RectBivariateSpline(
            np.arange(self.shape()[0]), np.arange(self.shape()[1]), rangeSpreadingLoss)
        return interpolator

    def noiseVectorMap(self, polarization, lutShift=False):
        ''' Convert noise vectors into full grid pixels '''
        # lutShift is introduced to correct erroneous range shifts in the annotated noise vectors.
        # lutShift is a new functionality developed after we published the reference, R1.
        #
        # The pattern of the annotated noise vectors are azimuth variant because the elevation
        # antenna pattern (EAP) correction in IPF processor uses averaged terrain height to account
        # for look angle changes along azimuth. This is correct approach, but the problem is, the
        # changes of EAP due to terrain height are actually 2-D, while the correction in IPF
        # considers azimuthal variation only (1-D). As a result, the EAP correction for the areas
        # where the terrain height changes largely in range direction does not work well.
        # As a first-aid fix for this, we suggest to shift the annotated noise vectors in range
        # direction by comparing the patterns of annotated noise vectors and the reproduced noise
        # vector-like gains used for EAP correction.
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        swathBounds = self.import_swathBounds(polarization)
        subswathIndexMap = self.subswathIndexMap(polarization)
        noiseVectorMap = np.ones(self.shape()) * np.nan
        # interpolate range vectors
        if lutShift:
            cPx = 150    # clip size of side pixels
            annotatedElevationAngleIntp = self.geolocationGridPointInterpolator(polarization,
                'elevationAngle')
            elevationAntennaPatternIntp = self.elevationAntennaPatternInterpolator(polarization)
            rangeSpreadingLossIntp = self.rangeSpreadingLossInterpolator(polarization)
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                swathBound = swathBounds[subswathID]
                line = np.array(noiseRangeVector['line'])
                valid = (   (line >= min(swathBound['firstAzimuthLine']))
                          * (line <= max(swathBound['lastAzimuthLine'])) )
                line = line[valid]
                pixel = np.array(noiseRangeVector['pixel'])[valid]
                noiseRangeLut = np.array(noiseRangeVector['noiseRangeLut'])[valid]
                xBins = np.arange(min(swathBound['firstRangeSample']),
                                  max(swathBound['lastRangeSample']) + 1)
                zi = np.ones((valid.sum(), len(xBins))) * np.nan
                for li,y in enumerate(line):
                    x = np.array(pixel[li])
                    z = np.array(noiseRangeLut[li])
                    valid = (subswathIndexMap[y,x]==iSW) * (z > 0)
                    valid[np.gradient(x)==0] = False
                    if valid.sum()==0:
                        continue
                    # simulate range dependent part of the combined gain in Eq.9-46 of the
                    # reference, R2.
                    referencePattern = ((1. / elevationAntennaPatternIntp(y, xBins[cPx:-cPx])
                        / rangeSpreadingLossIntp(y, xBins[cPx:-cPx]))**2).flatten()
                    zInterpolator = InterpolatedUnivariateSpline(x[valid], z[valid])
                    # Finding pixel shift, OPTION 1: Cross-correlation
                    # here we find only single value of shift per line per subswath
                    xShiftPixel, deltaShift, ni = 0, np.inf, 0
                    while deltaShift > 1e-2 and ni < 10:
                        ni += 1
                        noiseVector = zInterpolator(xBins + xShiftPixel)
                        deltaShift = est_shift(referencePattern, noiseVector)
                        xShiftPixel += deltaShift
                    '''
                    # Finding pixel shift, OPTION 2: Minimizing difference
                    errFunc = lambda x: np.sum((
                          (zInterpolator(xBins[cPx:-cPx]+x)/referencePattern)
                        - np.polyval(np.polyfit(xBins[cPx:-cPx],
                                                zInterpolator(xBins[cPx:-cPx]+x)/referencePattern,
                                                deg=1),
                                     xBins[cPx:-cPx]+x))**2)
                    xShiftPixel = fminbound(errFunc, -200, +200)
                    '''
                    #print(iSW, y, ni, xShiftPixel)
                    # here we find x shift for all pixels in one line in subswath ...
                    # via elevation angle increment
                    meanElevationAngleIncrement = np.mean(
                        np.diff(annotatedElevationAngleIntp(y,xBins)))
                    xShiftAngle = meanElevationAngleIncrement * xShiftPixel
                    xShiftPixel = np.squeeze( xShiftAngle
                        / np.gradient(np.squeeze(annotatedElevationAngleIntp(y,xBins))) )
                    zi[li,:] = InterpolatedUnivariateSpline(x[valid], z[valid])(xBins + xShiftPixel)
                valid = np.isfinite(np.sum(zi,axis=1))
                interpFunc = RectBivariateSpline(xBins, line[valid], zi[valid,:].T, kx=1, ky=1)
                valid = (subswathIndexMap==iSW)
                noiseVectorMap[valid] = interpFunc(np.arange(self.shape()[1]),
                                                   np.arange(self.shape()[0])).T[valid]
        else:
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                swathBound = swathBounds[subswathID]
                line = np.array(noiseRangeVector['line'])
                valid = (   (line >= min(swathBound['firstAzimuthLine']))
                          * (line <= max(swathBound['lastAzimuthLine'])) )
                line = line[valid]
                pixel = np.array(noiseRangeVector['pixel'])[valid]
                noiseRangeLut = np.array(noiseRangeVector['noiseRangeLut'])[valid]
                xBins = np.arange(min(swathBound['firstRangeSample']),
                                  max(swathBound['lastRangeSample']) + 1)
                zi = np.ones((valid.sum(), len(xBins))) * np.nan
                for li,y in enumerate(line):
                    x = np.array(pixel[li])
                    z = np.array(noiseRangeLut[li])
                    valid = (subswathIndexMap[y,x]==iSW) * (z > 0)
                    valid[np.gradient(x)==0] = False
                    if valid.sum()==0:
                        continue
                    zi[li,:] = InterpolatedUnivariateSpline(x[valid], z[valid])(xBins)
                valid = np.isfinite(np.sum(zi,axis=1))
                interpFunc = RectBivariateSpline(xBins, line[valid], zi[valid,:].T, kx=1, ky=1)
                valid = (subswathIndexMap==iSW)
                noiseVectorMap[valid] = interpFunc(np.arange(self.shape()[1]),
                                                   np.arange(self.shape()[0])).T[valid]
        # interpolate azimuth vectors
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfBlocks = len(noiseAzimuthVector[subswathID]['firstAzimuthLine'])
            for iBlk in range(numberOfBlocks):
                xs = noiseAzimuthVector[subswathID]['firstRangeSample'][iBlk]
                xe = noiseAzimuthVector[subswathID]['lastRangeSample'][iBlk]
                ys = noiseAzimuthVector[subswathID]['firstAzimuthLine'][iBlk]
                ye = noiseAzimuthVector[subswathID]['lastAzimuthLine'][iBlk]
                yBins = np.arange(ys, ye+1)
                y = noiseAzimuthVector[subswathID]['line'][iBlk]
                z = noiseAzimuthVector[subswathID]['noiseAzimuthLut'][iBlk]
                if not isinstance(y, list):
                    noiseVectorMap[yBins, xs:xe+1] *= z
                else:
                    noiseVectorMap[yBins, xs:xe+1] *= ( InterpolatedUnivariateSpline(
                        y, z, k=1)(yBins)[:,np.newaxis] * np.ones(xe-xs+1) )
        return noiseVectorMap

    def subswathIndexMap(self, polarization):
        ''' Convert subswath indices into full grid pixels '''
        subswathIndexMap = np.zeros(self.shape(), dtype=np.uint8)
        swathBounds = self.import_swathBounds(polarization)
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            swathBound = swathBounds['%s%s' % (self.obsMode, iSW)]
            zipped = zip(swathBound['firstAzimuthLine'],
                         swathBound['firstRangeSample'],
                         swathBound['lastAzimuthLine'],
                         swathBound['lastRangeSample'])
            for fal, frs, lal, lrs in zipped:
                subswathIndexMap[fal:lal+1,frs:lrs+1] = iSW
        return subswathIndexMap

    def rawSigma0Map(self, polarization):
        ''' Get sigma nought without noise power subtraction '''
        DN2 = np.power(self['DN_' + polarization].astype(np.uint32), 2)
        sigma0 = DN2 / np.power(self.calibrationVectorMap(polarization), 2)
        #sigma0[DN2==0] = np.nan
        return sigma0

    def rawNoiseEquivalentSigma0Map(self, polarization, lutShift=False):
        ''' Get annotated noise equivalent sigma nought '''
        noiseEquivalentSigma0 = (   self.noiseVectorMap(polarization, lutShift=lutShift)
                                  / np.power(self.calibrationVectorMap(polarization), 2) )
        # pre-scaling is needed for noise vectors when they have very low values
        if 10 * np.log10(np.nanmean(noiseEquivalentSigma0)) < -40:
            # values from S1A_AUX_CAL_V20150722T120000_G20151125T104733.SAFE
            noiseCalibrationFactor = {
                'IW1':59658.3803, 'IW2':52734.43872, 'IW3':59758.6889,
                'EW1':56065.87, 'EW2':56559.76, 'EW3':44956.39, 'EW4':46324.29, 'EW5':43505.68 }
            subswathIndexMap = self.subswathIndexMap(polarization)
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                valid = (subswathIndexMap==iSW)
                noiseEquivalentSigma0[valid] *= ( {'IW':474, 'EW':1087}[self.obsMode]**2
                    * noiseCalibrationFactor['%s%s' % (self.obsMode, iSW)] )
        return noiseEquivalentSigma0

    def incidenceAngleMap(self, polarization):
        ''' Get incidence angle '''
        interpolator = self.geolocationGridPointInterpolator(polarization, 'incidenceAngle')
        incidenceAngle = np.squeeze(interpolator(
            np.arange(self.shape()[0]), np.arange(self.shape()[1])))
        return incidenceAngle

    def scallopingGainMap(self, polarization):
        ''' Compute scalloping gains for full grid pixels

        Returns
        -------
        2d array with scalloping gain in each subswath

        '''
        # see section III.A of the reference, R1.
        subswathIndexMap = self.subswathIndexMap(polarization)
        scallopingGainMap = np.ones(self.shape()) * np.nan
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            # azimuth antenna element patterns (AAEP) lookup table for given subswath
            AAEP = self.import_azimuthAntennaElementPattern(polarization)[subswathID]
            gainAAEP = np.array(AAEP['azimuthAntennaElementPattern'])
            angleAAEP = ( np.arange(-(len(gainAAEP)//2), len(gainAAEP)//2+1)
                          * AAEP['azimuthAngleIncrement'] )
            # subswath range center pixel index
            subswathCenterSampleIndex = self.subswathCenterSampleIndex(polarization)[subswathID]
            # slant range time along subswath range center
            interpolator = self.geolocationGridPointInterpolator(polarization, 'slantRangeTime')
            slantRangeTime = np.squeeze(interpolator(
                np.arange(self.shape()[0]), subswathCenterSampleIndex))
            # relative azimuth time along subswath range center
            interpolator = self.geolocationGridPointInterpolator(polarization, 'azimuthTime')
            azimuthTime = np.squeeze(interpolator(
                np.arange(self.shape()[0]), subswathCenterSampleIndex))
            # Doppler rate induced by satellite motion
            motionDopplerRate = self.azimuthFmRateAtGivenTime(
                polarization, azimuthTime, slantRangeTime)
            # antenna steering rate
            antennaSteeringRate = np.deg2rad(ANTENNA_STEERING_RATE[subswathID])
            # satellite absolute velocity along subswath range center
            satelliteVelocity = np.linalg.norm(
                self.orbitAtGivenTime(polarization, azimuthTime)['velocityXYZ'], axis=1)
            # Doppler rate induced by TOPS steering of antenna
            steeringDopplerRate = 2 * satelliteVelocity / RADAR_WAVELENGTH * antennaSteeringRate
            # combined Doppler rate (net effect)
            combinedDopplerRate = ( motionDopplerRate * steeringDopplerRate
                                    / (motionDopplerRate - steeringDopplerRate) )
            # full burst length in zero-Doppler time
            fullBurstLength = self.focusedBurstLengthInTime(polarization)[subswathID]
            # zero-Doppler azimuth time at each burst start
            burstStartTime = np.array([
                (t-self.time_coverage_center).total_seconds()
                for t in self.import_antennaPattern(polarization)[subswathID]['azimuthTime'] ])
            # burst overlapping length
            burstOverlap = fullBurstLength - np.diff(burstStartTime)
            burstOverlap = np.hstack([burstOverlap[0], burstOverlap])
            # time correction
            burstStartTime += burstOverlap / 2.
            # if burst start time does not cover the full image,
            # add more sample points using the closest burst length
            while burstStartTime[0] > azimuthTime[0]:
                burstStartTime = np.hstack(
                    [burstStartTime[0] - np.diff(burstStartTime)[0], burstStartTime])
            while burstStartTime[-1] < azimuthTime[-1]:
                burstStartTime = np.hstack(
                    [burstStartTime, burstStartTime[-1] + np.diff(burstStartTime)[-1]])
            # convert azimuth time to burst time
            burstTime = np.copy(azimuthTime)
            for li in range(len(burstStartTime)-1):
                valid = (   (azimuthTime >= burstStartTime[li])
                          * (azimuthTime < burstStartTime[li+1]) )
                burstTime[valid] -= (burstStartTime[li] + burstStartTime[li+1]) / 2.
            # compute antenna steering angle for each burst time
            antennaSteeringAngle = np.rad2deg(
                RADAR_WAVELENGTH / (2 * satelliteVelocity)
                * combinedDopplerRate * burstTime )
            # compute scalloping gain for each burst time
            burstAAEP = np.interp(antennaSteeringAngle, angleAAEP, gainAAEP)
            scallopingGain = 1. / 10**(burstAAEP/10.)
            # assign computed scalloping gain into each subswath
            valid = (subswathIndexMap==iSW)
            scallopingGainMap[valid] = (
                scallopingGain[:,np.newaxis] * np.ones((1,self.shape()[1])))[valid]
        return scallopingGainMap

    def adaptiveNoiseScaling(self, sigma0, noiseEquivalentSigma0, subswathIndexMap,
                             extraScalingParameters, windowSize):
        ''' adaptive noise scaling compensating for local residual noise power

        this compenastaion should not be used in NERSC algorithm as it destoys the texture
        it can be used in ESA algorithm but then reslts wont be comparable to NAP
        '''
        # see section III.E of the reference, R1.
        weights = np.ones((windowSize,windowSize)) / windowSize**2.
        meanSigma0 = convolve(sigma0, weights, mode='constant', cval=0.0 )
        meanNEsigma0 = convolve(noiseEquivalentSigma0, weights, mode='constant', cval=0.0 )
        meanSWindex = convolve(subswathIndexMap, weights, mode='constant', cval=0.0 )
        SNR = 10 * np.log10(meanSigma0 / meanNEsigma0 - 1)
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            interpFunc = InterpolatedUnivariateSpline(extraScalingParameters['SNR'],
                extraScalingParameters['%s%s' % (self.obsMode, iSW)], k=3)
            valid = np.isfinite(SNR) * (meanSWindex==iSW)
            yInterp = interpFunc(SNR[valid])
            noiseEquivalentSigma0[valid] = noiseEquivalentSigma0[valid] * yInterp
        return noiseEquivalentSigma0

    def modifiedNoiseEquivalentSigma0Map(self, polarization, localNoisePowerCompensation=False):
        ''' Get noise power-scaled and interswath power-balanced noise equivalent sigma nought '''
        # see section III of the reference, R1.
        # raw noise-equivalent sigma nought
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization, lutShift=True)
        # apply scalloping gain to noise-equivalent sigma nought
        if self.IPFversion >= 2.5 and self.IPFversion < 2.9:
            noiseEquivalentSigma0 *= self.scallopingGainMap(polarization)
        # subswath index map
        subswathIndexMap = self.subswathIndexMap(polarization)
        # import coefficients
        noiseScalingParameters, powerBalancingParameters, extraScalingParameters = (
            self.import_denoisingCoefficients(polarization)[:3])
        # apply noise scaling and power balancing to noise-equivalent sigma nought
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            valid = (subswathIndexMap==iSW)
            noiseEquivalentSigma0[valid] *= noiseScalingParameters['%s%s' % (self.obsMode, iSW)]
            noiseEquivalentSigma0[valid] += powerBalancingParameters['%s%s' % (self.obsMode, iSW)]
        # apply extra noise scaling to compensate for local residual noise power
        if localNoisePowerCompensation and (polarization=='HV' or polarization=='VH'):
            sigma0 = self.rawSigma0Map(polarization)
            noiseEquivalentSigma0 = self.adaptiveNoiseScaling(
                sigma0, noiseEquivalentSigma0, subswathIndexMap, extraScalingParameters, 5)
        return noiseEquivalentSigma0


    def thermalNoiseRemoval(self, polarization, algorithm='NERSC',
            localNoisePowerCompensation=False, preserveTotalPower=False, returnNESZ=False):
        ''' Get denoised sigma nought

        Parameters
        ----------
        polarization :
        algorithm
        localNoisePowerCompensation : bool
            only for esa lagorithm: reduce power of positive signal when negative signal is clipped
            should always be false in NERSC algrotihm
        preserveTotalPower : bool
            subtract anomaly of thermal noize rather than full power
        returnNESZ : bool
            return array with NESZ in addition to denoised sigma0

        Returns
        -------
        sigma0 : ndarray
            2D array
        '''
        if algorithm not in ['ESA', 'NERSC']:
            raise ValueError('algorithm must be \'ESA\' or \'NERSC\'')
        if not isinstance(preserveTotalPower,bool):
            raise ValueError('preserveTotalPower must be True or False')
        # subswath index map
        subswathIndexMap = self.subswathIndexMap(polarization)
        # raw sigma nought
        rawSigma0 = self.rawSigma0Map(polarization)
        if algorithm=='ESA':
            # use annotated noise-equivalent sigma nought
            noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization)
        elif algorithm=='NERSC':
            # use modified noise-equivalent sigma nought
            noiseEquivalentSigma0 = self.modifiedNoiseEquivalentSigma0Map(
                polarization, localNoisePowerCompensation=False)
        # noise subtraction
        sigma0 = rawSigma0 - noiseEquivalentSigma0
        if preserveTotalPower:
            # add mean noise power of the entire scene back to the noise subtracted sigma nought
            sigma0 += np.nanmean(noiseEquivalentSigma0)
        if algorithm=='ESA':
            # ESA SNAP S1TBX-like implementation (zero-clipping) for pixels with negative power
            sigma0[sigma0 < 0] = 0
        elif algorithm=='NERSC':
            sigma0[rawSigma0==0] = np.nan
            sigma0[rawSigma0 < 1e-5] = np.nan
        if returnNESZ:
            # return both noise power and noise-power-subtracted sigma nought
            return noiseEquivalentSigma0, sigma0
        else:
            # return noise power subtracted sigma nought
            return sigma0

    def add_denoised_band(self, polarization):
        ''' Add denoised sigma nought to Nansat object as a band '''
        # all of these bands are not needed except for sigma0_denoised #
        if not self.has_band('subswath_indices'):
            self.add_band(
                self.subswathIndexMap(polarization),
                parameters={'name':'subswath_indices'})
        if not self.has_band('NEsigma0_%s' % polarization + '_raw'):
            self.add_band(
                self.rawNoiseEquivalentSigma0Map(polarization),
                parameters={'name':'NEsigma0_%s' % polarization + '_raw'})
        if not self.has_band('sigma0_%s' % polarization + '_raw'):
            self.add_band(
                self.rawSigma0Map(polarization),
                parameters={'name':'sigma0_%s' % polarization + '_raw'})
        if not self.has_band('NEsigma0_%s' % polarization):
            self.add_band(
                self.modifiedNoiseEquivalentSigma0Map(polarization, localNoisePowerCompensation=False),
                parameters={'name':'NEsigma0_%s' % polarization})
        if not self.has_band('sigma0_%s' % polarization + '_denoised'):
            self.add_band(
                self.thermalNoiseRemoval(polarization),
                parameters={'name':'sigma0_%s' % polarization + '_denoised'})

    def experiment_noiseScaling(self, polarization, numberOfLinesToAverage=1000):
        ''' Generate experimental data for noise scaling parameter optimization

        Note:
        Saves noise scaling coefficients
        5 variables, several subswaths, many values for each subswath for each subblock selected in
        azimuth direction. Sub blocks are <numberOfLinesToAverage> lines long

        '''
        # see section III.B of the reference, R1.
        cPx = {'IW':100, 'EW':25}[self.obsMode]    # clip size of side pixels, 1km
        subswathIndexMap = self.subswathIndexMap(polarization)
        landmask = self.landmask(skipGCP=4)
        sigma0 = self.rawSigma0Map(polarization)
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization, lutShift=True)
        if self.IPFversion >= 2.5 and self.IPFversion < 2.9:
            noiseEquivalentSigma0 *= self.scallopingGainMap(polarization)
        validLineIndices = np.argwhere(
            np.sum(subswathIndexMap!=0,axis=1)==self.shape()[1])
        blockBounds = np.arange(validLineIndices.min(), validLineIndices.max(),
                                numberOfLinesToAverage, dtype='uint')
        results = { '%s%s' % (self.obsMode, li):
                        { 'sigma0':[],
                          'noiseEquivalentSigma0':[],
                          'scalingFactor':[],
                          'correlationCoefficient':[],
                          'fitResidual':[] }
                    for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        results['IPFversion'] = self.IPFversion
        for iBlk in range(len(blockBounds)-1):
            if landmask[blockBounds[iBlk]:blockBounds[iBlk+1]].sum() != 0:
                continue
            # 2D matrix with data for each block
            blockS0 = sigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockN0 = noiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockSWI = subswathIndexMap[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            pixelValidity = (np.nanmean(blockS0 - blockN0 * 0.5, axis=0) > 0)
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                pixelIndex = np.nonzero((blockSWI==iSW).sum(axis=0) * pixelValidity)[0][cPx:-cPx]
                if pixelIndex.sum()==0:
                    continue
                # averaging in azimuth direction
                meanS0 = np.nanmean(np.where(blockSWI==iSW, blockS0, np.nan), axis=0)[pixelIndex]
                meanN0 = np.nanmean(np.where(blockSWI==iSW, blockN0, np.nan), axis=0)[pixelIndex]
                # weight is proportional to gradient of sigma0, the areas where s0 varies the most
                weight = abs(np.gradient(meanN0))
                weight = weight / weight.sum() * np.sqrt(len(weight))
                # polyfit is used only to return error (second return param) when fitting func:
                # s0-k*no = f(A*x + B). Where:
                # s0 - sigma0
                # n0 - thermal noise
                # k - noise scaling (to be identified at later stage of fitting)
                # x - pixel index
                # A, B - just some polynom coeffs that are not used
                errFunc = lambda k,x,s0,n0,w: np.polyfit(x,s0-k*n0,w=w,deg=1,full=True)[1].item()
                # now with polynom fitting function in place, K (the noise scaling coefficient)
                # is fitted iteratively using fminbound
                scalingFactor = fminbound(errFunc, 0, 3,
                    args=(pixelIndex,meanS0,meanN0,weight), disp=False).item()
                # correlatin between sigma0 and scaled noise
                correlationCoefficient = np.corrcoef(meanS0, scalingFactor * meanN0)[0,1]
                # error of fitting of the seclected scaling factor (K)
                fitResidual = np.polyfit(pixelIndex, meanS0 - scalingFactor * meanN0,
                                         w=weight, deg=1, full=True)[1].item()

                # NB1: in future triple fitting (polyfit for A,B; fminbound for K;
                # polyfit for fitResidual) can be replaced with scipy.optimize.curve_fit:
                # s0 = a + b*x + K*n0
                # NB2: Here we assume that s0 is linearly dependent on x. But this assumption is not
                # very correct. Especially for the first subswath. It looks more like exponential
                # decay and results become a bit overcorrected. Maybe linear function can be
                # replaced with more complex one but it needs data for training.
                results[subswathID]['sigma0'].append(meanS0)
                results[subswathID]['noiseEquivalentSigma0'].append(meanN0)
                results[subswathID]['scalingFactor'].append(scalingFactor)
                results[subswathID]['correlationCoefficient'].append(correlationCoefficient)
                results[subswathID]['fitResidual'].append(fitResidual)
        np.savez(self.name.split('.')[0] + '_noiseScaling.npz', **results)


    def experiment_powerBalancing(self, polarization, numberOfLinesToAverage=1000):
        ''' Generate experimental data for interswath power balancing parameter optimization '''
        # see section III.C of the reference, R1.
        cPx = {'IW':100, 'EW':25}[self.obsMode]    # clip size of side pixels, 1km
        subswathIndexMap = self.subswathIndexMap(polarization)
        landmask = self.landmask(skipGCP=4)
        sigma0 = self.rawSigma0Map(polarization)
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization, lutShift=True)
        if self.IPFversion >= 2.5 and self.IPFversion < 2.9:
            noiseEquivalentSigma0 *= self.scallopingGainMap(polarization)
        rawNoiseEquivalentSigma0 = noiseEquivalentSigma0.copy()
        noiseScalingParameters = self.import_denoisingCoefficients(polarization)[0]
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            valid = (subswathIndexMap==iSW)
            noiseEquivalentSigma0[valid] *= noiseScalingParameters['%s%s' % (self.obsMode, iSW)]
        validLineIndices = np.argwhere(
            np.sum(subswathIndexMap!=0,axis=1)==self.shape()[1])
        blockBounds = np.arange(validLineIndices.min(), validLineIndices.max(),
                                numberOfLinesToAverage, dtype='uint')
        results = { '%s%s' % (self.obsMode, li):
                        { 'sigma0':[],
                          'noiseEquivalentSigma0':[],
                          'balancingPower':[],
                          'correlationCoefficient':[],
                          'fitResidual':[] }
                    for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        results['IPFversion'] = self.IPFversion
        for iBlk in range(len(blockBounds)-1):
            if landmask[blockBounds[iBlk]:blockBounds[iBlk+1]].sum() != 0:
                continue
            blockS0 = sigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockN0 = noiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockRN0 = rawNoiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockSWI = subswathIndexMap[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            pixelValidity = (np.nanmean(blockS0 - blockRN0 * 0.5, axis=0) > 0)
            if pixelValidity.sum() <= (blockS0.shape[1] * 0.9):
                continue
            fitCoefficients = []
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                pixelIndex = np.nonzero((blockSWI==iSW).sum(axis=0) * pixelValidity)[0][cPx:-cPx]
                if pixelIndex.sum()==0:
                    continue
                meanS0 = np.nanmean(np.where(blockSWI==iSW, blockS0, np.nan), axis=0)[pixelIndex]
                meanN0 = np.nanmean(np.where(blockSWI==iSW, blockN0, np.nan), axis=0)[pixelIndex]
                meanRN0 = np.nanmean(np.where(blockSWI==iSW, blockRN0, np.nan), axis=0)[pixelIndex]
                fitResults = np.polyfit(pixelIndex, meanS0 - meanN0, deg=1, full=True)
                fitCoefficients.append(fitResults[0])
                results[subswathID]['sigma0'].append(meanS0)
                results[subswathID]['noiseEquivalentSigma0'].append(meanRN0)
                results[subswathID]['correlationCoefficient'].append(np.corrcoef(meanS0, meanN0)[0,1])
                results[subswathID]['fitResidual'].append(fitResults[1].item())
            balancingPower = np.zeros(5)
            for li in range(4):
                interswathBounds = ( np.where(np.gradient(blockSWI,axis=1)==0.5)[1]
                                     .reshape(4*numberOfLinesToAverage,2)[li::4].mean() )
                power1 = fitCoefficients[li][0] * interswathBounds + fitCoefficients[li][1]
                power2 = fitCoefficients[li+1][0] * interswathBounds + fitCoefficients[li+1][1]
                balancingPower[li+1] = power2 - power1
            balancingPower = np.cumsum(balancingPower)
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                valid = (blockSWI==iSW)
                blockN0[valid] += balancingPower[iSW-1]
            #powerBias = np.nanmean(blockRN0-blockN0)
            powerBias = np.nanmean((blockRN0-blockN0)[blockSWI>=2])
            balancingPower += powerBias
            blockN0 += powerBias
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                results['%s%s' % (self.obsMode, iSW)]['balancingPower'].append(balancingPower[iSW-1])
        np.savez(self.name.split('.')[0] + '_powerBalancing.npz', **results)

    def experiment_extraScaling(self, polarization):
        ''' Generate experimental data for extra scaling parameter optimization '''
        # see section III.E of the reference, R1.
        cPx = {'IW':1000, 'EW':250}[self.obsMode]    # clip size of side pixels, 10km
        nBins = 1001
        windowSizeMin = 3
        windowSizeMax = 27
        snnrRange = np.array([0, +4], dtype=np.float)
        snnrRange = [ snnrRange[0]-(snnrRange[-1]-snnrRange[0])/(nBins-1)/2.,
                      snnrRange[-1]+(snnrRange[-1]-snnrRange[0])/(nBins-1)/2. ]
        nnsdRange = np.array([0, +2], dtype=np.float)
        nnsdRange = [ nnsdRange[0]-(nnsdRange[-1]-nnsdRange[0])/(nBins-1)/2.,
                      nnsdRange[-1]+(nnsdRange[-1]-nnsdRange[0])/(nBins-1)/2. ]
        dBsnnrRange = np.array([-5, +5], dtype=np.float)
        dBsnnrRange = [ dBsnnrRange[0]-(dBsnnrRange[-1]-dBsnnrRange[0])/(nBins-1)/2.,
                        dBsnnrRange[-1]+(dBsnnrRange[-1]-dBsnnrRange[0])/(nBins-1)/2. ]
        esfRange = np.array([0, +10], dtype=np.float)
        esfRange = [ esfRange[0]-(esfRange[-1]-esfRange[0])/(nBins-1)/2.,
                     esfRange[-1]+(esfRange[-1]-esfRange[0])/(nBins-1)/2. ]
        subswathIndexMap = self.subswathIndexMap(polarization)
        noiseEquivalentSigma0, sigma0 = self.thermalNoiseRemoval(
            polarization, algorithm='NERSC', localNoisePowerCompensation=False,
            preserveTotalPower=False, returnNESZ=True )
        windowSizes = np.arange(windowSizeMin,windowSizeMax+1,2)
        results = { 'extraScalingFactorHistogram':
                        { '%s%s' % (self.obsMode, li):
                              np.zeros((len(windowSizes), nBins, nBins), dtype=np.int64)
                          for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1)},
                    'noiseNormalizedStandardDeviationHistogram':
                        { '%s%s' % (self.obsMode, li):
                              np.zeros((len(windowSizes), nBins, nBins), dtype=np.int64)
                          for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1)} }
        results['IPFversion'] = self.IPFversion
        results['windowSizes'] = windowSizes
        results['snnrEdges'] = np.linspace(snnrRange[0], snnrRange[-1], nBins+1)
        results['nnsdEdges'] = np.linspace(nnsdRange[0], nnsdRange[-1], nBins+1)
        results['dBsnnrEdges'] = np.linspace(dBsnnrRange[0], dBsnnrRange[-1], nBins+1)
        results['esfEdges'] = np.linspace(esfRange[0], esfRange[-1], nBins+1)
        for li, windowSize in enumerate(windowSizes):
            kernelMean = np.ones((windowSize,windowSize)) / windowSize**2
            meanNoise = convolve(noiseEquivalentSigma0, kernelMean)
            meanSignal = convolve(sigma0, kernelMean)
            meanSubswathIndexMap = convolve(subswathIndexMap.astype(np.float), kernelMean)
            signalPlusNoiseToNoiseRatio = (meanSignal + meanNoise) / meanNoise
            standardDeviation = np.sqrt(convolve(sigma0**2, kernelMean) - meanSignal**2)
            noiseNormalizedStandardDeviation = standardDeviation / meanNoise
            kernelSum = np.ones((windowSize,windowSize))
            numberOfPositives = convolve((sigma0 > 0).astype(np.float), kernelSum)
            sumSignal = meanSignal * windowSize**2
            sumNoise = meanNoise * windowSize**2
            sumZeroClippedSignal = convolve(np.where(sigma0 > 0, sigma0, 0), kernelSum)
            extraScalingFactor = 1 + ( (sumZeroClippedSignal-sumSignal) / sumNoise
                                       * (windowSize**2 / numberOfPositives) )
            extraScalingFactor[numberOfPositives==0] = np.nan
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                valid = (abs(meanSubswathIndexMap - iSW) < (1/windowSize/2))
                valid[:cPx,:] = False
                valid[-cPx:,:] = False
                valid[:,:cPx] = False
                valid[:,-cPx:] = False
                results['noiseNormalizedStandardDeviationHistogram'][subswathID][li] = np.histogram2d(
                    signalPlusNoiseToNoiseRatio[valid], noiseNormalizedStandardDeviation[valid],
                    bins=nBins, range=[snnrRange, nnsdRange])[0]
                results['extraScalingFactorHistogram'][subswathID][li] = np.histogram2d(
                    10*np.log10(signalPlusNoiseToNoiseRatio[valid]), extraScalingFactor[valid],
                    bins=nBins, range=[dBsnnrRange, esfRange])[0]
        np.savez_compressed(self.name.split('.')[0] + '_extraScaling.npz', **results)

    def landmask(self, skipGCP=4):
        ''' Generate landmask by reversing MODIS watermask data '''
        if skipGCP not in [1,2,4,5]:
            raise ValueError('skipGCP must be one of the following values: 1,2,4,5')
        originalGCPs = self.vrt.dataset.GetGCPs()
        numberOfGCPs = len(originalGCPs)
        index = np.arange(0,numberOfGCPs).reshape(numberOfGCPs//21,21)
        skipRowGCP = max([ y for y in range(1,numberOfGCPs//21)
                           if ((numberOfGCPs//21 -1) % y == 0) and y <= skipGCP ])
        sampledGCPs = [ originalGCPs[i] for i in np.concatenate(index[::skipRowGCP,::skipGCP]) ]
        projectionInfo = self.vrt.dataset.GetGCPProjection()
        dummy = self.vrt.dataset.SetGCPs(sampledGCPs, projectionInfo)
        landmask = (self.watermask(tps=True)[1]==2)
        dummy = self.vrt.dataset.SetGCPs(originalGCPs, projectionInfo)
        return landmask

    def texturalNoiseRemoval(self, polarization, preserveTotalPower=True, windowSize=15):
        """ Thermal noise removal followed by textural noise compensation

        In case of HH only jump between ew1 and ew2 is automatically compensated in the ew1.
        For HV coefficients for scaling standard deviation are loaded from pregenerated NPZ.

        Parameters
        ----------
        polarisation : str
            'HH' or 'HV'
        preserveTotalPower : bool
            Add mean total power to the denoised signal?
        windowSize : int
            Size of window where SD compenstaion is preformed

        Returns
        -------
        sigma0 : 2d numpy.ndarray
            Full size array with thermal and texture noise removed

        """
        if polarization=='HH':
            # for HH, apply simple adjustment of SDs between EW1 and EW2 for now...
            windowSize = windowSize*2       # use larger window size for HH
            sigma0 = self.rawSigma0Map(polarization)
            sigma0[sigma0==0] = np.nan
            offsetPower = 0
            nanMask = np.isnan(sigma0)
            subswathIndexMap = self.subswathIndexMap(polarization)
            # get data segments along subswath boundaries between EW1 and EW2
            swb = self.import_swathBounds(polarization)['%s1' % self.obsMode]
            sw1 = np.vstack([sigma0[fa:la+1,lr-windowSize+1:lr+1] for lr,fa,la in
                      zip(swb['lastRangeSample'],swb['firstAzimuthLine'],swb['lastAzimuthLine'])])
            sw2 = np.vstack([sigma0[fa:la+1,lr+1:lr+windowSize+1] for lr,fa,la in
                      zip(swb['lastRangeSample'],swb['firstAzimuthLine'],swb['lastAzimuthLine'])])
            # compute scaling factor from the ratios of EW1 and EW2 for mean and standard deviation
            meanRatio = [ np.nanmean(sw2[ri:ri+windowSize,:])/np.nanmean(sw1[ri:ri+windowSize,:])
                          for ri in range(len(sw1)-windowSize) ]
            stdRatio = [ np.nanstd(sw2[ri:ri+windowSize,:])/np.nanstd(sw1[ri:ri+windowSize,:])
                         for ri in range(len(sw1)-windowSize) ]
            scalingFactor = np.nanmedian(np.array(stdRatio) / np.array(meanRatio))
            #print('Computed extra SD scaling factor for %s1 = %f' % (self.obsMode, scalingFactor))
            if scalingFactor < 1.0 or scalingFactor > 1.3:
                scalingFactor = 1.15
            # apply scaling to EW1
            ri = np.where(subswathIndexMap==1)[1]
            riMin, riMax = ri.min(), ri.max()
            sswS0 = np.copy(sigma0[:,riMin:riMax+1])
            sswi = np.copy(subswathIndexMap[:,riMin:riMax+1])
            sswS0[sswi!=1] = np.nan
            sswS0 = fillGaps(sswS0)
            sswS0m = uniform_filter(sswS0, windowSize)
            sswS0 = (sswS0 - sswS0m) * scalingFactor + sswS0m
            sigma0[subswathIndexMap==1] = sswS0[sswi==1]
            sigma0[nanMask] = np.nan
        elif polarization=='HV':
            # see the reference paper R4
            # thermal noise removal
            noiseEquivalentSigma0, sigma0 = self.thermalNoiseRemoval(
                polarization, algorithm='NERSC', localNoisePowerCompensation=False,
                preserveTotalPower=False, returnNESZ=True )
            offsetPower = np.nanmean(noiseEquivalentSigma0)
            subswathIndexMap = self.subswathIndexMap(polarization)
            noiseVarianceParameters = self.import_denoisingCoefficients(polarization)[3]
            nanMask = np.isnan(sigma0)
            # subswath-wise processing
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                # rectangularize subswath data
                ai, ri = np.where(subswathIndexMap==iSW)
                aiMin, aiMax, riMin, riMax = ai.min(), ai.max(), ri.min(), ri.max()
                sswS0 = np.copy(sigma0[:,riMin:riMax+1])
                sswN0 = np.copy(noiseEquivalentSigma0[:,riMin:riMax+1])
                sswi = np.copy(subswathIndexMap[:,riMin:riMax+1])
                sswS0[sswi!=iSW] = np.nan
                sswN0[sswi!=iSW] = np.nan
                sswS0 = fillGaps(sswS0)
                sswN0 = fillGaps(sswN0)
                # compute local signal plus noise to noise ratio (SNNR)
                sswS0m = uniform_filter(sswS0, windowSize)
                sswN0m = uniform_filter(sswN0, windowSize)
                SNNR = (sswS0m + sswN0m) / sswN0m
                SNNR[SNNR<1] = 1
                SD = np.sqrt( uniform_filter(sswS0**2, windowSize) - sswS0m**2 )
                # simulate noise equivalent standard deviation (NESD)
                simSD = noiseVarianceParameters['%s%s' % (self.obsMode, iSW)] * sswN0m * (1./SNNR)
                # NESD compensation. See Eq.9 of the reference paper R4.
                sswS0 = (sswS0 - sswS0m) * (1. - simSD / SD) + sswS0m
                sigma0[subswathIndexMap==iSW] = sswS0[sswi==iSW]
            sigma0[nanMask] = np.nan
        if preserveTotalPower:
            sigma0 += offsetPower
        # mask out pixels where the raw DNs are not valid
        sigma0[self.rawSigma0Map(polarization)==0] = np.nan
        return sigma0
