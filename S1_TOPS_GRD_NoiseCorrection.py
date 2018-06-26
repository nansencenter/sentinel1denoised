import os, sys, glob, warnings, zipfile, requests, subprocess
import numpy as np
from datetime import datetime, timedelta
from xml.dom.minidom import parse, parseString
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.ndimage import convolve
from scipy.optimize import fminbound
from nansat import Nansat

warnings.simplefilter("ignore")


# define some constants
SPEED_OF_LIGHT = 299792458.
RADAR_FREQUENCY = 5.405000454334350e+09
RADAR_WAVELENGTH = SPEED_OF_LIGHT / RADAR_FREQUENCY
ANTENNA_STEERING_RATE = {
    'IW1': 1.590368784, 'IW2': 0.979863325, 'IW3': 1.397440818,
    'EW1': 2.390895448, 'EW2': 2.811502724, 'EW3': 2.366195855,
    'EW4': 2.512694636, 'EW5': 2.122855427 }


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
    return np.polynomial.hermite.hermval(xi, np.polynomial.hermite.hermfit(x,y,deg=3))


def range_to_target(satPosVec, lookVec, terrainHeight=0):
    ''' Compute slant range distance to target on WGS-84 Earth ellipsoid '''
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
    ''' planar rotation about a given axis '''
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



class Sentinel1Image(Nansat):

    def __init__(self, filename, mapperName='sentinel1_l1', logLevel=30):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__( self, filename,
                         mapperName=mapperName, logLevel=logLevel)
        if ( self.fileName.split('/')[-1][:16]
             not in ['S1A_IW_GRDH_1SDH', 'S1A_IW_GRDH_1SDV',
                     'S1B_IW_GRDH_1SDH', 'S1B_IW_GRDH_1SDV',
                     'S1A_EW_GRDM_1SDH', 'S1A_EW_GRDM_1SDV',
                     'S1B_EW_GRDM_1SDH', 'S1B_EW_GRDM_1SDV' ] ):
             raise ValueError( 'Source file must be Sentinel-1A/1B '
                 'IW_GRDH_1SDH, IW_GRDH_1SDV, EW_GRDM_1SDH, or EW_GRDM_1SDV product.' )
        self.platform = self.fileName.split('/')[-1][:3]
        self.obsMode = self.fileName.split('/')[-1][4:6]
        txPol = self.fileName.split('/')[-1][15]
        self.annotationXML = {}
        self.calibrationXML = {}
        self.noiseXML = {}
        if zipfile.is_zipfile(self.filename):
            zf = zipfile.PyZipFile(self.filename)
            annotationFiles = [fn for fn in zf.namelist() if 'annotation/s1' in fn]
            calibrationFiles = [fn for fn in zf.namelist() if 'annotation/calibration/calibration-s1' in fn]
            noiseFiles = [fn for fn in zf.namelist() if 'annotation/calibration/noise-s1' in fn]
            for polarization in [txPol + 'H', txPol + 'V']:
                self.annotationXML[polarization] = parseString(
                    [zf.read(fn) for fn in annotationFiles if polarization.lower() in fn][0])
                self.calibrationXML[polarization] = parseString(
                    [zf.read(fn) for fn in calibrationFiles if polarization.lower() in fn][0])
                self.noiseXML[polarization] = parseString(
                    [zf.read(fn) for fn in noiseFiles if polarization.lower() in fn][0])
            self.manifestXML = parseString(zf.read([fn for fn in zf.namelist() if 'manifest.safe' in fn][0]))
            zf.close()
        else:
            annotationFiles = [fn for fn in glob.glob(self.filename+'/annotation/*') if 's1' in fn]
            calibrationFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*') if 'calibration-s1' in fn]
            noiseFiles = [fn for fn in glob.glob(self.filename+'/annotation/calibration/*') if 'noise-s1' in fn]
            for polarization in [txPol + 'H', txPol + 'V']:
                self.annotationXML[polarization] = parseString(
                    [open(fn).read() for fn in annotationFiles if polarization.lower() in fn][0])
                self.calibrationXML[polarization] = parseString(
                    [open(fn).read() for fn in calibrationFiles if polarization.lower() in fn][0])
                self.noiseXML[polarization] = parseString(
                    [open(fn).read() for fn in noiseFiles if polarization.lower() in fn][0])
            self.manifestXML = parseString(open(glob.glob(self.filename+'/manifest.safe')[0]).read())
        self.time_coverage_center = ( self.time_coverage_start + timedelta(
            seconds=(self.time_coverage_end - self.time_coverage_start).total_seconds()/2) )
        self.IPFversion = float(self.manifestXML.getElementsByTagName('safe:software')[0]
                                .attributes['version'].value)
        if self.IPFversion < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Noise correction cannot be achieved using this module. '
                  'Denoising vectors in annotation file are not qualified.\n')
            return
        elif 2.43 <= self.IPFversion < 2.53:
            print('\nWARNING: IPF version of input image is lower than 2.53! '
                  'Noise correction result can be wrong.\n')
        resourceList = self.manifestXML.getElementsByTagName('resource')
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
            cal_url = 'https://qc.sentinel1.eo.esa.int/product/%s/%s_%s/%s/%s.TGZ' %(parts[0], parts[1], parts[2], parts[3][1:], filename)
            r = requests.get(cal_url, stream=True)
            with open(cal_file_tgz, "wb") as f:
                f.write(r.content)
            subprocess.call(['tar', '-xzvf', cal_file_tgz, '-C', self.aux_data_dir])
        self.auxiliaryCalibration_file = cal_file


    def import_antennaPattern(self, polarization):
        ''' import antenna pattern information from annotation XML DOM '''
        antennaPatternList = self.annotationXML[polarization].getElementsByTagName('antennaPattern')[1:]
        antennaPattern = { '%s%s' % (self.obsMode, li):
            { 'azimuthTime':[], 'slantRangeTime':[], 'elevationAngle':[],
              'elevationPattern':[], 'incidenceAngle':[], 'terrainHeight':[], 'roll':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in antennaPatternList:
            swath = get_DOM_nodeValue(iList,['swath'])
            antennaPattern[swath]['azimuthTime'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
            antennaPattern[swath]['slantRangeTime'].append(
                get_DOM_nodeValue(iList,['slantRangeTime'],'float'))
            antennaPattern[swath]['elevationAngle'].append(
                get_DOM_nodeValue(iList,['elevationAngle'],'float'))
            antennaPattern[swath]['elevationPattern'].append(
                get_DOM_nodeValue(iList,['elevationPattern'],'float'))
            antennaPattern[swath]['incidenceAngle'].append(
                get_DOM_nodeValue(iList,['incidenceAngle'],'float'))
            antennaPattern[swath]['terrainHeight'].append(
                get_DOM_nodeValue(iList,['terrainHeight'],'float'))
            antennaPattern[swath]['roll'].append(
                get_DOM_nodeValue(iList,['roll'],'float'))
        return antennaPattern


    def import_azimuthAntennaElementPattern(self, polarization):
        ''' import azimuth antenna element pattern information from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        azimuthAntennaElementPattern = { '%s%s' % (self.obsMode, li):
            { 'azimuthAngleIncrement':[], 'azimuthAntennaElementPattern':[],
              'absoluteCalibrationConstant':[], 'noiseCalibrationFactor':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            if ( swath in azimuthAntennaElementPattern.keys()
                 and get_DOM_nodeValue(iList,['polarisation'])==polarization ):
                elem = iList.getElementsByTagName('azimuthAntennaElementPattern')[0]
                azimuthAntennaElementPattern[swath]['azimuthAngleIncrement'] = (
                    get_DOM_nodeValue(elem,['azimuthAngleIncrement'],'float') )
                azimuthAntennaElementPattern[swath]['azimuthAntennaElementPattern'] = (
                    get_DOM_nodeValue(elem,['values'],'float') )
                azimuthAntennaElementPattern[swath]['absoluteCalibrationConstant'] = (
                    get_DOM_nodeValue(iList,['absoluteCalibrationConstant'],'float') )
                azimuthAntennaElementPattern[swath]['noiseCalibrationFactor'] = (
                    get_DOM_nodeValue(iList,['noiseCalibrationFactor'],'float') )
        return azimuthAntennaElementPattern


    def import_azimuthAntennaPattern(self, polarization):
        ''' import azimuth antenna pattern information from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        azimuthAntennaPattern = { '%s%s' % (self.obsMode, li):
            { 'azimuthAngleIncrement':[], 'azimuthAntennaPattern':[],
              'absoluteCalibrationConstant':[], 'noiseCalibrationFactor':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            if ( swath in azimuthAntennaPattern.keys()
                 and get_DOM_nodeValue(iList,['polarisation'])==polarization ):
                elem = iList.getElementsByTagName('azimuthAntennaPattern')[0]
                azimuthAntennaPattern[swath]['azimuthAngleIncrement'] = (
                    get_DOM_nodeValue(elem,['azimuthAngleIncrement'],'float') )
                azimuthAntennaPattern[swath]['azimuthAntennaPattern'] = (
                    get_DOM_nodeValue(elem,['values'],'float') )
                azimuthAntennaPattern[swath]['absoluteCalibrationConstant'] = (
                    get_DOM_nodeValue(iList,['absoluteCalibrationConstant'],'float') )
                azimuthAntennaPattern[swath]['noiseCalibrationFactor'] = (
                    get_DOM_nodeValue(iList,['noiseCalibrationFactor'],'float') )
        return azimuthAntennaPattern


    def import_azimuthFmRate(self, polarization):
        ''' import azimuth frequency modulation rate information from annotation XML DOM '''
        azimuthFmRateList = self.annotationXML[polarization].getElementsByTagName('azimuthFmRate')
        azimuthFmRate = { 'azimuthTime':[], 't0':[], 'azimuthFmRatePolynomial':[] }
        for iList in azimuthFmRateList:
            azimuthFmRate['azimuthTime'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
            azimuthFmRate['t0'].append(
                    get_DOM_nodeValue(iList,['t0'],'float'))
            azimuthFmRate['azimuthFmRatePolynomial'].append(
                    get_DOM_nodeValue(iList,['azimuthFmRatePolynomial'],'float'))
        return azimuthFmRate


    def import_calibrationVector(self, polarization):
        ''' import calibration vectors from calibration annotation XML DOM '''
        calibrationVectorList = self.calibrationXML[polarization].getElementsByTagName('calibrationVector')
        calibrationVector = { 'azimuthTime':[], 'line':[], 'pixel':[],
            'sigmaNought':[], 'betaNought':[], 'gamma':[], 'dn':[] }
        for iList in calibrationVectorList:
            calibrationVector['azimuthTime'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
            calibrationVector['line'].append(
                get_DOM_nodeValue(iList,['line'],'int'))
            calibrationVector['pixel'].append(
                get_DOM_nodeValue(iList,['pixel'],'int'))
            calibrationVector['sigmaNought'].append(
                get_DOM_nodeValue(iList,['sigmaNought'],'float'))
            calibrationVector['betaNought'].append(
                get_DOM_nodeValue(iList,['betaNought'],'float'))
            calibrationVector['gamma'].append(
                get_DOM_nodeValue(iList,['gamma'],'float'))
            calibrationVector['dn'].append(
                get_DOM_nodeValue(iList,['dn'],'float'))
        return calibrationVector


    def import_denoisingCoefficients(self, polarization):
        ''' import denoising coefficients '''
        satID = self.fileName.split('/')[-1][:3]
        denoisingParameters = np.load( os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'denoising_parameters_%s.npz' % satID),
            encoding='latin1' )
        noiseScalingParameters = {}
        powerBalancingParameters = {}
        extraScalingParameters = {}
        extraScalingParameters['SNR'] = []
        noiseVarianceParameters = {}
        versionsInLUT = list(denoisingParameters['noiseScalingParameters'].item()['%s1' % self.obsMode].keys())
        closestIPFversion = float(
            versionsInLUT[np.argmin(abs(self.IPFversion - np.array(versionsInLUT, dtype=np.float)))])
        IPFversion = float(self.IPFversion)
        sensingDate = datetime.strptime(self.fileName.split('/')[-1].split('_')[4], '%Y%m%dT%H%M%S')
        if satID=='S1B' and IPFversion==2.72 and sensingDate >= datetime(2017,1,16,13,42,34):
            # Adaption for special case.
            # ESA abrubtly changed scaling LUT in AUX_PP1 from 20170116 while keeping the IPFversion.
            # After this change, the scaling parameters seems be much closer to those of IPFv 2.8.
            IPFversion = 2.8
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            if 'noiseScalingParameters' in denoisingParameters.keys():
                try:
                    noiseScalingParameters[subswathID] = (
                        denoisingParameters['noiseScalingParameters'].item()
                        [subswathID]['%.1f' % IPFversion] )
                except:
                    print('WARNING: noise scaling parameters for IPF version %s are missing.\n'
                          '         parameters for IPF version %s will be used for now.'
                          % (IPFversion, closestIPFversion))
                    noiseScalingParameters[subswathID] = (
                        denoisingParameters['noiseScalingParameters'].item()
                        [subswathID]['%.1f' % closestIPFversion] )
            else:
                noiseScalingParameters[subswathID] = 1.0
            if 'powerBalancingParameters' in denoisingParameters.keys():
                try:
                    powerBalancingParameters[subswathID] = (
                        denoisingParameters['powerBalancingParameters'].item()
                        [subswathID]['%.1f' % IPFversion] )
                except:
                    print('WARNING: power balancing parameters for IPF version %s are missing.\n'
                          '         parameters for IPF version %s will be used for now.'
                          % (IPFversion, closestIPFversion))
                    powerBalancingParameters[subswathID] = (
                        denoisingParameters['powerBalancingParameters'].item()
                        [subswathID]['%.1f' % closestIPFversion] )
            else:
                powerBalancingParameters[subswathID] = 0.0
            if 'extraScalingParameters' in denoisingParameters.keys():
                extraScalingParameters[subswathID] = np.array(
                    denoisingParameters['extraScalingParameters'].item()[subswathID] )
                extraScalingParameters['SNR'] = np.array(
                    denoisingParameters['extraScalingParameters'].item()['SNNR'] )
            else:
                extraScalingParameters['SNNR'] = np.linspace(-30,+30,601)
                extraScalingParameters[subswathID] = np.ones(601)
            if 'noiseVarianceParameters' in denoisingParameters.keys():
                noiseVarianceParameters[subswathID] = (
                    denoisingParameters['noiseVarianceParameters'].item()[subswathID] )
            else:
                noiseVarianceParameters[subswathID] = 0.0
        return noiseScalingParameters, powerBalancingParameters, extraScalingParameters, noiseVarianceParameters


    def import_elevationAntennaPattern(self, polarization):
        ''' import elevation antenna pattern information from auxiliary calibration XML DOM '''
        calParamsList = self.auxiliaryCalibrationXML.getElementsByTagName('calibrationParams')
        elevationAntennaPattern = { '%s%s' % (self.obsMode, li):
            { 'elevationAngleIncrement':[], 'elevationAntennaPattern':[],
              'absoluteCalibrationConstant':[], 'noiseCalibrationFactor':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList in calParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            if ( swath in elevationAntennaPattern.keys()
                 and get_DOM_nodeValue(iList,['polarisation'])==polarization ):
                elem = iList.getElementsByTagName('elevationAntennaPattern')[0]
                elevationAntennaPattern[swath]['elevationAngleIncrement'] = (
                    get_DOM_nodeValue(elem,['elevationAngleIncrement'],'float') )
                elevationAntennaPattern[swath]['elevationAntennaPattern'] = (
                    get_DOM_nodeValue(elem,['values'],'float') )
                elevationAntennaPattern[swath]['absoluteCalibrationConstant'] = (
                    get_DOM_nodeValue(iList,['absoluteCalibrationConstant'],'float') )
                elevationAntennaPattern[swath]['noiseCalibrationFactor'] = (
                    get_DOM_nodeValue(iList,['noiseCalibrationFactor'],'float') )
        return elevationAntennaPattern


    def import_geolocationGridPoint(self, polarization):
        ''' import geolocation grid point information from annotation XML DOM '''
        geolocationGridPointList = self.annotationXML[polarization].getElementsByTagName('geolocationGridPoint')
        geolocationGridPoint = { 'azimuthTime':[], 'slantRangeTime':[],
            'line':[], 'pixel':[], 'latitude':[], 'longitude':[], 'height':[],
            'incidenceAngle':[], 'elevationAngle':[] }
        for iList in geolocationGridPointList:
            geolocationGridPoint['azimuthTime'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
            geolocationGridPoint['slantRangeTime'].append(
                get_DOM_nodeValue(iList,['slantRangeTime'],'float'))
            geolocationGridPoint['line'].append(
                get_DOM_nodeValue(iList,['line'],'int'))
            geolocationGridPoint['pixel'].append(
                get_DOM_nodeValue(iList,['pixel'],'int'))
            geolocationGridPoint['latitude'].append(
                get_DOM_nodeValue(iList,['latitude'],'float'))
            geolocationGridPoint['longitude'].append(
                get_DOM_nodeValue(iList,['longitude'],'float'))
            geolocationGridPoint['height'].append(
                get_DOM_nodeValue(iList,['height'],'float'))
            geolocationGridPoint['incidenceAngle'].append(
                get_DOM_nodeValue(iList,['incidenceAngle'],'float'))
            geolocationGridPoint['elevationAngle'].append(
                get_DOM_nodeValue(iList,['elevationAngle'],'float'))
        return geolocationGridPoint


    def import_noiseVector(self, polarization):
        ''' import noise vectors from noise annotation XML DOM '''
        noiseRangeVector = { 'azimuthTime':[], 'line':[], 'pixel':[], 'noiseRangeLut':[] }
        noiseAzimuthVector = { '%s%s' % (self.obsMode, li):
            { 'firstAzimuthLine':[], 'firstRangeSample':[],
              'lastAzimuthLine':[], 'lastRangeSample':[], 'line':[], 'noiseAzimuthLut':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        if self.IPFversion < 2.9:
            noiseVectorList = self.noiseXML[polarization].getElementsByTagName('noiseVector')
            for iList in noiseVectorList:
                noiseRangeVector['azimuthTime'].append(
                    datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
                noiseRangeVector['line'].append(
                    get_DOM_nodeValue(iList,['line'],'int'))
                noiseRangeVector['pixel'].append(
                    get_DOM_nodeValue(iList,['pixel'],'int'))
                noiseRangeVector['noiseRangeLut'].append(
                    get_DOM_nodeValue(iList,['noiseLut'],'float'))
        elif self.IPFversion >= 2.9:
            noiseRangeVectorList = self.noiseXML[polarization].getElementsByTagName('noiseRangeVector')
            for iList in noiseRangeVectorList:
                noiseRangeVector['azimuthTime'].append(
                    datetime.strptime(get_DOM_nodeValue(iList,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
                noiseRangeVector['line'].append(
                    get_DOM_nodeValue(iList,['line'],'int'))
                noiseRangeVector['pixel'].append(
                    get_DOM_nodeValue(iList,['pixel'],'int'))
                noiseRangeVector['noiseRangeLut'].append(
                    get_DOM_nodeValue(iList,['noiseRangeLut'],'float'))
            noiseAzimuthVectorList = self.noiseXML[polarization].getElementsByTagName('noiseAzimuthVector')
            for iList in noiseAzimuthVectorList:
                swath = get_DOM_nodeValue(iList,['swath'],'str')
                noiseAzimuthVector[swath]['firstAzimuthLine'].append(
                    get_DOM_nodeValue(iList,['firstAzimuthLine'],'int'))
                noiseAzimuthVector[swath]['firstRangeSample'].append(
                    get_DOM_nodeValue(iList,['firstRangeSample'],'int'))
                noiseAzimuthVector[swath]['lastAzimuthLine'].append(
                    get_DOM_nodeValue(iList,['lastAzimuthLine'],'int'))
                noiseAzimuthVector[swath]['lastRangeSample'].append(
                    get_DOM_nodeValue(iList,['lastRangeSample'],'int'))
                noiseAzimuthVector[swath]['line'].append(
                    get_DOM_nodeValue(iList,['line'],'int'))
                noiseAzimuthVector[swath]['noiseAzimuthLut'].append(
                    get_DOM_nodeValue(iList,['noiseAzimuthLut'],'float'))
        return noiseRangeVector, noiseAzimuthVector


    def import_orbit(self, polarization):
        ''' import orbit information from annotation XML DOM '''
        orbitList = self.annotationXML[polarization].getElementsByTagName('orbit')
        orbit = { 'time':[], 'position':{'x':[], 'y':[], 'z':[]},
                  'velocity':{'x':[], 'y':[], 'z':[]} }
        for iList in orbitList:
            orbit['time'].append(
                datetime.strptime(get_DOM_nodeValue(iList,['time']), '%Y-%m-%dT%H:%M:%S.%f'))
            orbit['position']['x'].append(
                get_DOM_nodeValue(iList,['position','x'],'float'))
            orbit['position']['y'].append(
                get_DOM_nodeValue(iList,['position','y'],'float'))
            orbit['position']['z'].append(
                get_DOM_nodeValue(iList,['position','z'],'float'))
            orbit['velocity']['x'].append(
                get_DOM_nodeValue(iList,['velocity','x'],'float'))
            orbit['velocity']['y'].append(
                get_DOM_nodeValue(iList,['velocity','y'],'float'))
            orbit['velocity']['z'].append(
                get_DOM_nodeValue(iList,['velocity','z'],'float'))
        return orbit


    def import_processorScalingFactor(self, polarization):
        ''' import swath processing scaling factors from annotation XML DOM '''
        swathProcParamsList = self.annotationXML[polarization].getElementsByTagName('swathProcParams')
        processorScalingFactor = {}
        for iList in swathProcParamsList:
            swath = get_DOM_nodeValue(iList,['swath'])
            processorScalingFactor[swath] = get_DOM_nodeValue(iList,['processorScalingFactor'],'float')
        return processorScalingFactor


    def import_swathBounds(self, polarization):
        ''' import swath bounds information from annotation XML DOM '''
        swathMergeList = self.annotationXML[polarization].getElementsByTagName('swathMerge')
        swathBounds = { '%s%s' % (self.obsMode, li):
            { 'azimuthTime':[], 'firstAzimuthLine':[],
              'firstRangeSample':[], 'lastAzimuthLine':[], 'lastRangeSample':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        for iList1 in swathMergeList:
            swath = get_DOM_nodeValue(iList1,['swath'])
            swathBoundsList = iList1.getElementsByTagName('swathBounds')
            for iList2 in swathBoundsList:
                swathBounds[swath]['azimuthTime'].append(
                    datetime.strptime(get_DOM_nodeValue(iList2,['azimuthTime']), '%Y-%m-%dT%H:%M:%S.%f'))
                swathBounds[swath]['firstAzimuthLine'].append(
                    get_DOM_nodeValue(iList2,['firstAzimuthLine'],'int'))
                swathBounds[swath]['firstRangeSample'].append(
                    get_DOM_nodeValue(iList2,['firstRangeSample'],'int'))
                swathBounds[swath]['lastAzimuthLine'].append(
                    get_DOM_nodeValue(iList2,['lastAzimuthLine'],'int'))
                swathBounds[swath]['lastRangeSample'].append(
                    get_DOM_nodeValue(iList2,['lastRangeSample'],'int'))
        return swathBounds


    def azimuthFmRateAtGivenTime(self, polarization, relativeAzimuthTime, slantRangeTime):
        ''' interpolate azimuth frequency modulation rate for given time vectors '''
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
        ''' focused burst length in zero-Doppler time domain '''
        azimuthTimeIntevalInSLC = 1. / get_DOM_nodeValue(self.annotationXML[polarization],['azimuthFrequency'],'float')
        inputDimensionsList = self.annotationXML[polarization].getElementsByTagName('inputDimensions')
        focusedBurstLengthInTime = {}
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
        ''' generate interpolator for items in geolocation grid point list '''
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
        ''' interpolate orbit parameters for given time vector '''
        stateVectors = self.import_orbit(polarization)
        orbitTime = np.array([ (t-self.time_coverage_center).total_seconds()
                                for t in stateVectors['time'] ])
        orbitAtGivenTime = { 'relativeAzimuthTime':relativeAzimuthTime,
            'positionXYZ':[], 'velocityXYZ':[] }
        for t in relativeAzimuthTime:
            useIndices = sorted(np.argsort(abs(orbitTime-t))[:4])
            orbitAtGivenTime['positionXYZ'].append([
                cubic_hermite_interpolation( orbitTime[useIndices],
                    np.array(stateVectors['position'][component])[useIndices], t)
                for component in ['x','y','z'] ])
            orbitAtGivenTime['velocityXYZ'].append([
                cubic_hermite_interpolation( orbitTime[useIndices],
                    np.array(stateVectors['velocity'][component])[useIndices], t)
                for component in ['x','y','z'] ])
        orbitAtGivenTime['positionXYZ'] = np.squeeze(orbitAtGivenTime['positionXYZ'])
        orbitAtGivenTime['velocityXYZ'] = np.squeeze(orbitAtGivenTime['velocityXYZ'])
        return orbitAtGivenTime


    def subswathCenterSampleIndex(self, polarization):
        ''' range center pixel indices along azimuth for each subswath '''
        swathBounds = self.import_swathBounds(polarization)
        subswathCenterSampleIndex = {}
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfLines = ( np.array(swathBounds[subswathID]['lastAzimuthLine'])
                              - np.array(swathBounds[subswathID]['firstAzimuthLine']) + 1 )
            midPixelIndices = ( np.array(swathBounds[subswathID]['firstRangeSample'])
                                + np.array(swathBounds[subswathID]['lastRangeSample']) ) / 2.
            subswathCenterSampleIndex[subswathID] = int(round(
                np.sum(midPixelIndices * numberOfLines) / np.sum(numberOfLines) ))
        return subswathCenterSampleIndex


    def calibrationVectorMap(self, polarization):
        ''' convert calibration vectors into full grid pixels '''
        calibrationVector = self.import_calibrationVector(polarization)
        swathBounds = self.import_swathBounds(polarization)
        subswathIndexMap = self.subswathIndexMap(polarization)
        calibrationVectorMap = np.ones(self.shape()) * np.nan
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


    def elevationAngleInterpolator(self, polarization):
        ''' generate elevation angle interpolator using the refined elevation angle
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
            slantRangeDistance = (299792458. / 2 * slantRangeTimeIntp(line[li], pixel[li])).item()
            lookVector = np.cross(velocityVector, positionVector)
            lookVector /= np.linalg.norm(lookVector)
            rotationAxis = np.cross(positionVector, lookVector)
            rotationAxis /= np.linalg.norm(rotationAxis)
            depressionAngle = 45.
            nIter = 0
            status = False
            while not (status or (nIter >= maxIter)):
                rotatedLookVector1 = planar_rotation(rotationAxis, lookVector, depressionAngle)
                rotatedLookVector2 = planar_rotation(rotationAxis, lookVector, depressionAngle + angleStep)
                err1 = range_to_target(positionVector, rotatedLookVector1) - slantRangeDistance
                err2 = range_to_target(positionVector, rotatedLookVector2) - slantRangeDistance
                depressionAngle += ( err1 / ((err1 - err2) / angleStep) )
                status = np.abs(err1).max() < distanceThreshold
                nIter += 1
            elevationAngle.append(90. - depressionAngle)
        elevationAngle = np.reshape(elevationAngle, (len(uniqueLine), len(uniquePixel)))
        interpolator = RectBivariateSpline(uniqueLine, uniquePixel, elevationAngle)
        return interpolator


    def rollAngleInterpolator(self, polarization):
        antennaPattern = self.import_antennaPattern(polarization)
        relativeAzimuthTime = []
        rollAngle = []
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            subswathID = '%s%s' % (self.obsMode, iSW)
            relativeAzimuthTime.append([ (t-self.time_coverage_center).total_seconds()
                                         for t in antennaPattern[subswathID]['azimuthTime'] ])
            rollAngle.append(antennaPattern[subswathID]['roll'])
        relativeAzimuthTime = np.hstack(relativeAzimuthTime)
        rollAngle = np.hstack(rollAngle)
        sortIndex = np.argsort(rollAngle)
        interpolator = InterpolatedUnivariateSpline(relativeAzimuthTime[sortIndex], rollAngle[sortIndex])
        return interpolator


    def noiseVectorMap(self, polarization, lutShift=False):
        ''' convert noise vectors into full grid pixels '''
        noiseRangeVector, noiseAzimuthVector = self.import_noiseVector(polarization)
        swathBounds = self.import_swathBounds(polarization)
        subswathIndexMap = self.subswathIndexMap(polarization)
        noiseVectorMap = np.ones(self.shape()) * np.nan
        annoElevAngleIntp = self.geolocationGridPointInterpolator(polarization, 'elevationAngle')
        refinedElevAngleIntp = self.elevationAngleInterpolator(polarization)
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
                if valid.sum()==0:
                    continue
                # noiseLut shifting
                if lutShift:
                    xShift = np.squeeze( (annoElevAngleIntp(y,xBins) - refinedElevAngleIntp(y,xBins))
                                          / np.gradient(np.squeeze(annoElevAngleIntp(y,xBins))) )
                    zi[li,:] = InterpolatedUnivariateSpline(x[valid], z[valid])(xBins + xShift)
                else:
                    zi[li,:] = InterpolatedUnivariateSpline(x[valid], z[valid])(xBins)
            valid = np.isfinite(np.sum(zi,axis=1))
            interpFunc = RectBivariateSpline(xBins, line[valid], zi[valid,:].T, kx=1, ky=1)
            valid = (subswathIndexMap==iSW)
            noiseVectorMap[valid] = interpFunc(np.arange(self.shape()[1]),
                                               np.arange(self.shape()[0])).T[valid]
        if self.IPFversion >= 2.9:
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
                        noiseVectorMap[yBins, xs:xe+1] *= \
                            InterpolatedUnivariateSpline(y, z, k=1)(yBins)[:,np.newaxis] * np.ones(xe-xs+1)
        return noiseVectorMap


    def subswathIndexMap(self, polarization):
        ''' convert subswath indices into full grid pixels '''
        subswathIndexMap = np.zeros(self.shape(), dtype=np.uint8)
        swathBounds = self.import_swathBounds(polarization)
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            swathBound = swathBounds['%s%s' % (self.obsMode, iSW)]
            zipped = zip(swathBound['firstAzimuthLine'], swathBound['firstRangeSample'],
                         swathBound['lastAzimuthLine'], swathBound['lastRangeSample'])
            for fal, frs, lal, lrs in zipped:
                subswathIndexMap[fal:lal+1,frs:lrs+1] = iSW
        return subswathIndexMap


    def rawSigma0Map(self, polarization):
        ''' noise power unsubtracted sigma nought '''
        DN2 = np.power(self['DN_' + polarization].astype(np.uint32), 2)
        sigma0 = DN2 / np.power(self.calibrationVectorMap(polarization), 2)
        #sigma0[DN2==0] = np.nan
        return sigma0


    def rawNoiseEquivalentSigma0Map(self, polarization):
        ''' original annotated noise equivalent sigma nought '''
        noiseEquivalentSigma0 = (   self.noiseVectorMap(polarization)
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
                noiseEquivalentSigma0[valid] *= (
                      noiseCalibrationFactor['%s%s' % (self.obsMode, iSW)]
                    * {'IW':474, 'EW':1087}[self.obsMode]**2 )
        return noiseEquivalentSigma0


    def incidenceAngleMap(self, polarization):
        ''' incidence angle '''
        interpolator = self.geolocationGridPointInterpolator(polarization, 'incidenceAngle')
        incidenceAngle = np.squeeze(interpolator(
            np.arange(self.shape()[0]), np.arange(self.shape()[1])))
        return incidenceAngle


    def scallopingGainMap(self, polarization):
        ''' scalloping gain of full grid pixels '''
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
            motionDopplerRate = self.azimuthFmRateAtGivenTime(polarization, azimuthTime, slantRangeTime)
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
                scallopingGain[:,np.newaxis] * np.ones((1,self.shape()[1])) )[valid]
        return scallopingGainMap


    def adaptiveNoiseScaling(self, sigma0, noiseEquivalentSigma0, subswathIndexMap,
                             extraScalingParameters, windowSize):
        ''' adaptive noise scaling for compensating local residual noise power '''
        meanSigma0 = convolve(
            sigma0, np.ones((windowSize,windowSize)) / windowSize**2.,
            mode='constant', cval=0.0 )
        meanNEsigma0 = convolve(
            noiseEquivalentSigma0, np.ones((windowSize,windowSize)) / windowSize**2.,
            mode='constant', cval=0.0 )
        meanSWindex = convolve(
            subswathIndexMap, np.ones((windowSize,windowSize)) / windowSize**2.,
            mode='constant', cval=0.0 )
        SNR = 10 * np.log10(meanSigma0 / meanNEsigma0 - 1)
        for iSW in range(1,6):
            interpFunc = InterpolatedUnivariateSpline(
                             extraScalingParameters['SNR'],
                             extraScalingParameters['%s%s' % (self.obsMode, iSW)], k=3)
            valid = np.isfinite(SNR) * (meanSWindex==iSW)
            yInterp = interpFunc(SNR[valid])
            noiseEquivalentSigma0[valid] = noiseEquivalentSigma0[valid] * yInterp
        return noiseEquivalentSigma0


    def modifiedNoiseEquivalentSigma0Map(self, polarization, localNoisePowerCompensation=False):
        ''' scaled and balanced noise equivalent sigma nought for cross-polarization channgel '''
        # raw noise-equivalent sigma nought
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization)
        meanNESZraw = np.nanmean(noiseEquivalentSigma0)
        # apply scalloping gain to noise-equivalent sigma nought
        if self.IPFversion >= 2.5 and self.IPFversion < 2.9:
            noiseEquivalentSigma0 *= self.scallopingGainMap(polarization)
        # subswath index map
        subswathIndexMap = self.subswathIndexMap(polarization)
        # import coefficients
        noiseScalingParameters, powerBalancingParameters, extraScalingParameters = (
            self.import_denoisingCoefficients(polarization)[:3] )
        # apply noise scaling and power balancing to noise-equivalent sigma nought
        for iSW in range(1,6):
            valid = (subswathIndexMap==iSW)
            noiseEquivalentSigma0[valid] *= noiseScalingParameters['%s%s' % (self.obsMode, iSW)]
            noiseEquivalentSigma0[valid] += powerBalancingParameters['%s%s' % (self.obsMode, iSW)]
        meanNESZmodified = np.nanmean(noiseEquivalentSigma0)
        # total noise power should be preserved
        #noiseEquivalentSigma0 += (meanNESZraw - meanNESZmodified)
        # apply extra noise scaling for compensating local residual noise power
        if localNoisePowerCompensation and (polarization=='HV' or polarization=='VH'):
            sigma0 = self.rawSigma0Map(polarization)
            noiseEquivalentSigma0 = self.adaptiveNoiseScaling(
                sigma0, noiseEquivalentSigma0, subswathIndexMap,
                extraScalingParameters, 5 )
        return noiseEquivalentSigma0


    def thermalNoiseRemoval(self, polarization, algorithm='NERSC',
            localNoisePowerCompensation=False, preserveTotalPower=False, returnNESZ=False):
        ''' thermal noise removal for operational use '''
        if algorithm not in ['ESA', 'NERSC']:
            raise ValueError('algorithm must be \'ESA\' or \'NERSC\'')
        if not isinstance(preserveTotalPower,bool):
            raise ValueError('preserveTotalPower must be True or False')
        # subswath index map
        subswathIndexMap = self.subswathIndexMap(polarization)
        # raw sigma nought
        rawSigma0 = self.rawSigma0Map(polarization)
        if algorithm=='ESA':
            # use raw noise-equivalent sigma nought
            noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization)
        elif algorithm=='NERSC':
            # modified noise-equivalent sigma nought
            noiseEquivalentSigma0 = self.modifiedNoiseEquivalentSigma0Map(
                polarization, localNoisePowerCompensation=localNoisePowerCompensation)
        # noise subtraction
        sigma0 = rawSigma0 - noiseEquivalentSigma0
        if preserveTotalPower:
            # add mean noise power back to the noise subtracted sigma nought
            sigma0 += np.nanmean(noiseEquivalentSigma0)
        if algorithm=='ESA':
            # ESA SNAP S1TBX-like implementation (zero-clipping) for pixels with negative power
            sigma0[sigma0 < 0] = 0
        if returnNESZ:
            # return both noise power and noise-power-subtracted sigma nought
            return noiseEquivalentSigma0, sigma0
        else:
            # return noise power subtracted sigma nought
            return sigma0


    '''
    def thermalNoiseRemoval_dev(self, polarization,
            preserveTotalPower=True, returnNESZ=False, windowSize=25):
        noiseEquivalentSigma0, sigma0 = self.thermalNoiseRemoval(
            polarization, algorithm='NERSC', localNoisePowerCompensation=False,
            preserveTotalPower=False, returnNESZ=True )
        if returnNESZ:
            NESZ = np.copy(noiseEquivalentSigma0)
        offsetPower = np.nanmean(noiseEquivalentSigma0)
        subswathIndexMap = self.subswathIndexMap(polarization)
        numberOfAzimuthBlocks = self.shape()[0] // windowSize + bool(self.shape()[0] % windowSize)
        numberOfRangeBlocks = self.shape()[1] // windowSize + bool(self.shape()[1] % windowSize)
        sigma0 = np.pad(sigma0,
            ((0,numberOfAzimuthBlocks*windowSize-self.shape()[0]),
             (0,numberOfRangeBlocks*windowSize-self.shape()[1])),
            'constant', constant_values=np.nan)
        sigma0 = [
            sigma0[ri*windowSize:(ri+1)*windowSize,
                   ci*windowSize:(ci+1)*windowSize]
            for (ri,ci) in np.ndindex(numberOfAzimuthBlocks, numberOfRangeBlocks) ]
        noiseEquivalentSigma0 = np.pad(noiseEquivalentSigma0,
            ((0,numberOfAzimuthBlocks*windowSize-self.shape()[0]),
             (0,numberOfRangeBlocks*windowSize-self.shape()[1])),
            'constant', constant_values=np.nan)
        noiseEquivalentSigma0 = [
            noiseEquivalentSigma0[ri*windowSize:(ri+1)*windowSize,
                                  ci*windowSize:(ci+1)*windowSize]
            for (ri,ci) in np.ndindex(numberOfAzimuthBlocks, numberOfRangeBlocks) ]
        subswathIndexMap = np.pad(subswathIndexMap,
            ((0,numberOfAzimuthBlocks*windowSize-self.shape()[0]),
             (0,numberOfRangeBlocks*windowSize-self.shape()[1])),
            'constant', constant_values=np.nan)
        subswathIndexMap = [
            subswathIndexMap[ri*windowSize:(ri+1)*windowSize,
                             ci*windowSize:(ci+1)*windowSize]
            for (ri,ci) in np.ndindex(numberOfAzimuthBlocks, numberOfRangeBlocks) ]
        noiseVarianceParameters = self.import_denoisingCoefficients(polarization)[3]
        for li in range(numberOfAzimuthBlocks * numberOfRangeBlocks):
            uniqueIndices = np.unique(subswathIndexMap[li])
            uniqueIndices = uniqueIndices[uniqueIndices>0]
            for ui in uniqueIndices:
                valid = (subswathIndexMap[li]==ui) * np.isfinite(sigma0[li])
                mS0 = np.mean(sigma0[li][valid])
                mNES0 = np.mean(noiseEquivalentSigma0[li][valid])
                SNNR = (mS0 + mNES0) / mNES0
                STD = np.std(sigma0[li][valid])
                simSTD = noiseVarianceParameters['%s%s' % (self.obsMode, ui)] * mNES0 * (1./SNNR)
                if simSTD >= STD:
                    #simSTD = STD * (1-np.finfo(STD).eps)
                    simSTD = STD - (simSTD - STD)
                if polarization=='HH' and self.obsMode=='EW':
                    if ui==1:
                        sigma0[li][valid] = (sigma0[li][valid] - mS0) * (1.-simSTD/STD) * 1.15 + mS0
                else:
                    sigma0[li][valid] = (sigma0[li][valid] - mS0) * (1.-simSTD/STD) + mS0
        sigma0 = np.concatenate(np.array_split(np.concatenate(sigma0, axis=1),
            numberOfAzimuthBlocks, axis=1 ), axis=0 )[:self.shape()[0],:self.shape()[1]]
        if preserveTotalPower:
            sigma0 += offsetPower
        if returnNESZ:
            # return both noise power and noise-power-subtracted sigma nought
            return NESZ, sigma0
        else:
            # return noise power subtracted sigma nought
            return sigma0
    '''


    def thermalNoiseRemoval_dev(self, polarization,
            preserveTotalPower=True, returnNESZ=False, windowSize=25):
        ''' thermal noise removal under development '''
        noiseEquivalentSigma0, sigma0 = self.thermalNoiseRemoval(
            polarization, algorithm='NERSC', localNoisePowerCompensation=False,
            preserveTotalPower=False, returnNESZ=True )
        if returnNESZ:
            NESZ = np.copy(noiseEquivalentSigma0)
        offsetPower = np.nanmean(noiseEquivalentSigma0)
        subswathIndexMap = self.subswathIndexMap(polarization)
        noiseVarianceParameters = self.import_denoisingCoefficients(polarization)[3]
        for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
            ri = np.where(subswathIndexMap==iSW)[1]
            riMin, riMax = ri.min(), ri.max()
            sswS0 = np.copy(sigma0[:,riMin:riMax+1])
            sswN0 = np.copy(noiseEquivalentSigma0[:,riMin:riMax+1])
            sswi = np.copy(subswathIndexMap[:,riMin:riMax+1])
            sswS0[sswi!=iSW] = np.nan
            sswN0[sswi!=iSW] = np.nan
            nr = sswS0.shape[1]
            for li in range(sswS0.shape[0]):
                fi = np.where(np.isfinite(sswS0[li]))[0]
                if len(fi)==0:
                    continue
                fiMin, fiMax = fi.min(), fi.max()
                if fiMax!=(nr-1):
                    sswS0[li][fiMax+1:nr] = sswS0[li][fiMax:fiMax-(nr-fiMax)+1:-1]
                    sswN0[li][fiMax+1:nr] = sswN0[li][fiMax:fiMax-(nr-fiMax)+1:-1]
                if fiMin!=0:
                    sswS0[li][0:fiMin] = sswS0[li][fiMin:fiMin+fiMin]
                    sswN0[li][0:fiMin] = sswN0[li][fiMin:fiMin+fiMin]
            sswS0m = convolve(sswS0, np.ones((windowSize,windowSize)) / windowSize**2, mode='reflect')
            sswN0m = convolve(sswN0, np.ones((windowSize,windowSize)) / windowSize**2, mode='reflect')
            SNNR = (sswS0m + sswN0m) / sswN0m
            SD = np.sqrt( convolve(sswS0**2, np.ones((windowSize,windowSize)) / windowSize**2, mode='reflect')
                          - (sswS0m*(windowSize**2))**2 / windowSize**4 )
            simSD = noiseVarianceParameters['%s%s' % (self.obsMode, iSW)] * sswN0m * (1./SNNR)
            if polarization=='HH' and self.obsMode=='EW' and iSW==1:
                sswS0 = (sswS0 - sswS0m) * (1. - simSD / SD) * 1.15 + sswS0m
            else:
                sswS0 = (sswS0 - sswS0m) * (1. - simSD / SD) + sswS0m
            sigma0[subswathIndexMap==iSW] = sswS0[sswi==iSW]
        if preserveTotalPower:
            sigma0 += offsetPower
        if returnNESZ:
            # return both noise power and noise-power-subtracted sigma nought
            return NESZ, sigma0
        else:
            # return noise power subtracted sigma nought
            return sigma0


    def add_denoised_band(self, polarization):
        if not self.has_band('subswath_indices'):
            self.add_band(self.subswathIndexMap(polarization),
                          parameters={'name': 'subswath_indices'})
        self.add_band(self.rawSigma0Map(polarization),
                      parameters={'name': 'sigma0_%s' % polarization + '_raw'})
        self.add_band(self.rawNoiseEquivalentSigma0Map(polarization),
                      parameters={'name': 'NEsigma0_%s' % polarization + '_raw'})
        self.add_band(self.modifiedNoiseEquivalentSigma0Map(polarization, localNoisePowerCompensation=False),
                      parameters={'name': 'NEsigma0_%s' % polarization})
        denoisedBandArray = self.thermalNoiseRemoval_dev(polarization)
        self.add_band(denoisedBandArray,
                      parameters={'name': 'sigma0_%s' % polarization + '_denoised'})


    def angularDependencyCorrection(self, polarization, sigma0, slope=-0.25):
        ''' compensate incidency angle dependency in sigma nought '''
        angularDependency = np.power(10, -slope * (self.incidenceAngleMap(polarization) - 20.0) / 10)
        return sigma0 * angularDependency


    def experiment_noiseScaling(self, polarization, numberOfLinesToAverage=1000):
        ''' generate experimental data for noise scaling parameter optimization '''
        clipSidePixels = {'IW':100, 'EW':25}[self.obsMode]      # 1km
        subswathIndexMap = self.subswathIndexMap(polarization)
        landmask = self.landmask(skipGCP=4)
        sigma0 = self.rawSigma0Map(polarization)
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization)
        if self.IPFversion >= 2.5 and self.IPFversion < 2.9:
            noiseEquivalentSigma0 *= self.scallopingGainMap(polarization)
        validLineIndices = np.argwhere(
            np.sum(subswathIndexMap!=0,axis=1)==self.shape()[1])
        blockBounds = np.arange(validLineIndices.min(), validLineIndices.max(),
                                numberOfLinesToAverage, dtype='uint')
        results = { '%s%s' % (self.obsMode, li):
            { 'sigma0':[], 'noiseEquivalentSigma0':[],
              'scalingFactor':[], 'correlationCoefficient':[], 'fitResidual':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        results['IPFversion'] = self.IPFversion
        for iBlk in range(len(blockBounds)-1):
            if landmask[blockBounds[iBlk]:blockBounds[iBlk+1]].sum() != 0:
                continue
            blockS0 = sigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockN0 = noiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockSWI = subswathIndexMap[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            pixelValidity = (np.nanmean(blockS0 - blockN0 * 0.5, axis=0) > 0)
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                pixelIndex = np.nonzero((blockSWI==iSW).sum(axis=0) * pixelValidity)[0][clipSidePixels:-clipSidePixels]
                if pixelIndex.sum()==0:
                    continue
                meanS0 = np.nanmean(np.where(blockSWI==iSW, blockS0, np.nan), axis=0)[pixelIndex]
                meanN0 = np.nanmean(np.where(blockSWI==iSW, blockN0, np.nan), axis=0)[pixelIndex]
                weight = abs(np.gradient(meanN0))
                weight = weight / weight.sum() * np.sqrt(len(weight))
                errorFunction = lambda k, x, s0, n0, w : np.polyfit(x, s0 - k * n0, w=w, deg=1, full=True)[1].item()
                scalingFactor = fminbound(errorFunction, 0, 3,
                    args=(pixelIndex,meanS0,meanN0,weight), disp=False).item()
                correlationCoefficient = np.corrcoef(meanS0, scalingFactor * meanN0)[0,1]
                fitResidual = np.polyfit(pixelIndex, meanS0 - scalingFactor * meanN0,
                                         w=weight, deg=1, full=True)[1].item()
                results[subswathID]['sigma0'].append(meanS0)
                results[subswathID]['noiseEquivalentSigma0'].append(meanN0)
                results[subswathID]['scalingFactor'].append(scalingFactor)
                results[subswathID]['correlationCoefficient'].append(correlationCoefficient)
                results[subswathID]['fitResidual'].append(fitResidual)
        np.savez(self.name.split('.')[0] + '_noiseScaling.npz', **results)


    def experiment_powerBalancing(self, polarization, numberOfLinesToAverage=1000):
        ''' generate experimental data for interswath power balancing parameter optimization '''
        clipSidePixels = {'IW':100, 'EW':25}[self.obsMode]      # 1km
        subswathIndexMap = self.subswathIndexMap(polarization)
        landmask = self.landmask(skipGCP=4)
        sigma0 = self.rawSigma0Map(polarization)
        noiseEquivalentSigma0 = self.rawNoiseEquivalentSigma0Map(polarization)
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
            { 'sigma0':[], 'noiseEquivalentSigma0':[],
              'balancingPower':[], 'correlationCoefficient':[], 'fitResidual':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        results['IPFversion'] = self.IPFversion
        for iBlk in range(len(blockBounds)-1):
            if landmask[blockBounds[iBlk]:blockBounds[iBlk+1]].sum() != 0:
                continue
            blockS0 = sigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockN0 = noiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockRawN0 = rawNoiseEquivalentSigma0[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            blockSWI = subswathIndexMap[blockBounds[iBlk]:blockBounds[iBlk+1],:]
            pixelValidity = (np.nanmean(blockS0 - blockRawN0 * 0.5, axis=0) > 0)
            if pixelValidity.sum() <= (blockS0.shape[1] * 0.9):
                continue
            fitCoefficients = []
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                subswathID = '%s%s' % (self.obsMode, iSW)
                pixelIndex = np.nonzero((blockSWI==iSW).sum(axis=0) * pixelValidity)[0][clipSidePixels:-clipSidePixels]
                if pixelIndex.sum()==0:
                    continue
                meanS0 = np.nanmean(np.where(blockSWI==iSW, blockS0, np.nan), axis=0)[pixelIndex]
                meanN0 = np.nanmean(np.where(blockSWI==iSW, blockN0, np.nan), axis=0)[pixelIndex]
                meanRawN0 = np.nanmean(np.where(blockSWI==iSW, blockRawN0, np.nan), axis=0)[pixelIndex]
                #weight = abs(np.gradient(meanN0))
                #weight = weight / weight.sum() * np.sqrt(len(weight))
                #fitResults = np.polyfit(pixelIndex, meanS0 - meanN0, w=weight, deg=1, full=True)
                fitResults = np.polyfit(pixelIndex, meanS0 - meanN0, deg=1, full=True)
                fitCoefficients.append(fitResults[0])
                results[subswathID]['sigma0'].append(meanS0)
                results[subswathID]['noiseEquivalentSigma0'].append(meanRawN0)
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
            #powerBias = np.nanmean(blockRawN0-blockN0)
            powerBias = np.nanmean((blockRawN0-blockN0)[blockSWI>=2])
            balancingPower += powerBias
            blockN0 += powerBias
            for iSW in range(1, {'IW':3, 'EW':5}[self.obsMode]+1):
                results['%s%s' % (self.obsMode, iSW)]['balancingPower'].append(balancingPower[iSW-1])
        np.savez(self.name.split('.')[0] + '_powerBalancing.npz', **results)


    def experiment_extraScaling(self, polarization, windowSize=25):
        ''' generate experimental data for extra scaling parameter optimization '''
        clipSidePixels = {'IW':1000, 'EW':250}[self.obsMode]      # 10km
        subswathIndexMap = self.subswathIndexMap(polarization)
        noiseEquivalentSigma0, sigma0 = self.thermalNoiseRemoval(
            polarization, algorithm='NERSC', localNoisePowerCompensation=False,
            preserveTotalPower=False, returnNESZ=True )
        numberOfAzimuthBlocks = self.shape()[0] / windowSize
        numberOfRangeBlocks = (self.shape()[1] - 2*clipSidePixels) / windowSize
        numberOfBlocks = numberOfAzimuthBlocks * numberOfRangeBlocks
        sigma0 = sigma0[
            :numberOfAzimuthBlocks*windowSize,
            clipSidePixels:clipSidePixels+numberOfRangeBlocks*windowSize]
        sigma0 = [ sigma0[ri*windowSize:(ri+1)*windowSize,
                          ci*windowSize:(ci+1)*windowSize]
                   for (ri,ci) in np.ndindex(numberOfAzimuthBlocks,
                                             numberOfRangeBlocks)  ]
        noiseEquivalentSigma0 = noiseEquivalentSigma0[
            :numberOfAzimuthBlocks*windowSize,
            clipSidePixels:clipSidePixels+numberOfRangeBlocks*windowSize]
        noiseEquivalentSigma0 = [
            noiseEquivalentSigma0[ri*windowSize:(ri+1)*windowSize,
                                  ci*windowSize:(ci+1)*windowSize]
            for (ri,ci) in np.ndindex(numberOfAzimuthBlocks,
                                      numberOfRangeBlocks)  ]
        subswathIndexMap = subswathIndexMap[
            :numberOfAzimuthBlocks*windowSize,
            clipSidePixels:clipSidePixels+numberOfRangeBlocks*windowSize]
        subswathIndexMap = [
            subswathIndexMap[ri*windowSize:(ri+1)*windowSize,
                             ci*windowSize:(ci+1)*windowSize]
            for (ri,ci) in np.ndindex(numberOfAzimuthBlocks,
                                      numberOfRangeBlocks)  ]
        results = { '%s%s' % (self.obsMode, li):
            { 'extraScalingFactor':[], 'signalPlusNoiseToNoiseRatio':[],
              'noiseNormalizedStandardDeviation':[] }
            for li in range(1, {'IW':3, 'EW':5}[self.obsMode]+1) }
        results['IPFversion'] = self.IPFversion
        zipped = zip(sigma0, noiseEquivalentSigma0, subswathIndexMap)
        for s0, n0, ind in zipped:
            uniqueIndex = np.unique(ind)
            if uniqueIndex.size!=1 or uniqueIndex==0:
                continue
            iSW = uniqueIndex.item()
            subswathID = '%s%s' % (self.obsMode, iSW)
            numberOfPositives = (s0 > 0).sum()
            zeroClipped = np.where(s0 >= 0, s0, 0)
            if numberOfPositives:
                alpha = 1 + ( np.nansum(zeroClipped-s0) / np.nansum(n0)
                              * (windowSize**2 / float(numberOfPositives)) )
            else:
                alpha = np.nan
            results[subswathID]['extraScalingFactor'].append(alpha)
            results[subswathID]['signalPlusNoiseToNoiseRatio'].append(np.nanmean(s0+n0) / np.nanmean(n0))
            results[subswathID]['noiseNormalizedStandardDeviation'].append(np.nanstd(s0) / np.nanmean(n0))
        np.savez(self.name.split('.')[0] + '_extraScaling.npz', **results)


    def landmask(self, skipGCP=4):
        ''' generate landmask by reversing MODIS watermask data '''
        if skipGCP not in [1,2,4,5]:
            raise ValueError('skipGCP must be one of the following values: 1,2,4,5')
        originalGCPs = self.vrt.dataset.GetGCPs()
        numberOfGCPs = len(originalGCPs)
        index = np.arange(0,numberOfGCPs).reshape(numberOfGCPs//21,21)
        skipRowGCP = max([ y for y in range(1,numberOfGCPs//21)
                           if ((numberOfGCPs//21 -1) % y == 0) and y <= skipGCP ])
        sampledGCPs = [ originalGCPs[i] for i in np.concatenate(index[::skipRowGCP,::skipGCP]) ]
        projectionInfo = self.vrt.dataset.GetGCPProjection()
        dummy = self.vrt.dataset.SetGCPs(sampledGCPs,projectionInfo)
        landmask = (self.watermask(tps=True)[1]==2)
        dummy = self.vrt.dataset.SetGCPs(originalGCPs,projectionInfo)
        return landmask

