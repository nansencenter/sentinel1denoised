import os
import glob
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString

import numpy as np
from scipy.interpolate import griddata, InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline

from nansat import Nansat

def getElem(elem, tags):
    ''' Get sub-element from XML element based on tags '''
    iElem = elem
    for iTag in tags:
        iElem = iElem.getElementsByTagName(iTag)[0]
    return iElem

def getValue(elem, tags):
    ''' Get value of XML subelement based on tags '''
    return getElem(elem, tags).childNodes[0].nodeValue

def convertTime2Sec(time):
    iTime = time
    HHMMSS = time.split('T')[1].split(':')
    secOfDay = float(HHMMSS[0])*3600 + float(HHMMSS[1])*60 + float(HHMMSS[2])
    return secOfDay

class Sentinel1Image(Nansat):
    """
    RADIOMETRIC CALIBRATION AND NOISE REMOVAL FOR S-1 GRD PRODUCT

    FOR HH CHANNEL,
        THERMAL NOISE SUBTRACTION + SCALOPING CORRECTION
        + ANGULAR DEPENDENCY REMOVAL (REFERECE ANGLE = 17.0 DEGREE)
        EMPIRICAL EQUATION: 0.60717 * exp(-0.12296X) + 0.02218
    FOR HV CHANNEL,
        THERMAL NOISE SUBTRACTION + SCALOPING CORRECTION
        + CONSTANT POWER ADDITION (-23.5 dB)

    HOW TO COMPUTE EXACT ZERO DOPPLER TIME, ZDT? THIS IS SOMEWHAT UNCLEAR YET.
    I INTRODUCED zdtBias TO ADJUST IT APPROXIMATELY.
    """
    def __init__(self, fileName, mapperName='', logLevel=30):
        ''' Read calibration/annotation XML files and AntennaElementPattern'''
        Nansat.__init__(self, fileName, mapperName=mapperName, logLevel=logLevel)

        self.calibXML = {}
        self.annotXML = {}

        for pol in ['HH', 'HV']:
            self.annotXML[pol] = parse(glob.glob(
                '%s/annotation/s1a*-%s-*.xml' % (self.fileName,
                                                 pol.lower()))[0])
            self.calibXML[pol] = {}
            for prod in ['calibration', 'noise']:
                self.calibXML[pol][prod] = parse(glob.glob(
                '%s/annotation/calibration/%s-*-%s-*.xml' % (self.fileName,
                                                             prod,
                                                             pol.lower()))[0])

        self.azimuthAntennaElementPattern = np.load(os.path.join(
                                os.path.dirname(os.path.realpath(__file__)),
                                'AAEP_V20150722.npz'))


    def get_calibration_LUT(self, pol, iProd):
        ''' Read calibration LUT from XML for a given polarization
        Parameters
        ----------
        pol : str
            polarisation: 'HH' or 'HV'
        iProd : str
            product: 'calibration' or 'noise'

        Returns
        -------
        oLUT : dict
            values, pixels, lines - 2D matrices

        '''
        if iProd not in ['calibration', 'noise']:
            raise ValueError('iProd must be calibration or noise')
        productDict = { 'calibration':'sigmaNought', 'noise':'noiseLut' }

        pixels = []
        lines = []
        values = []
        vectorList = getElem(self.calibXML[pol][iProd], [iProd + 'VectorList'])
        for iVector in vectorList.getElementsByTagName(iProd+'Vector'):
            pixels.append(map(int, getValue(iVector,['pixel']).split()))
            lines.append(int(getValue(iVector,['line'])))
            values.append(map(float, getValue(iVector,
                                              [productDict[iProd]]).split()))

        return dict(pixels = np.array(pixels),
                    lines = np.array(lines),
                    values = np.array(values))

    def get_swath_bounds(self, pol):
        ''' Get list of left right top bottom edges for blocks in each swath

        Parameters
        ----------
        pol : polarisation: 'HH' or 'HV'

        Returns
        -------
        swathBounds : dict
            values of first/last line/sample in multilevel dict:
            swathID
                firstAzimuthLine
                firstRangeSample
                lastAzimuthLine
                lastRangeSample
        '''
        keys = ['firstAzimuthLine', 'firstRangeSample',
                'lastAzimuthLine', 'lastRangeSample']
        swathMergeList = getElem(self.annotXML[pol], ['swathMergeList'])
        swathBounds = {}
        for iSwathMerge in swathMergeList.getElementsByTagName('swathMerge'):
            swathID = getValue(iSwathMerge, ['swath'])
            swathBoundsList = getElem(iSwathMerge, ['swathBoundsList'])
            swathBounds[swathID] = dict([(key,[]) for key in keys])
            for iSwathBounds in swathBoundsList.getElementsByTagName('swathBounds'):
                for key in keys:
                    swathBounds[swathID][key].append(
                                            int(getValue(iSwathBounds, [key])))
        return swathBounds

    def interpolate_lut(self, iLUT, bounds):
        ''' Interpolate noise or calibration lut to single full resolution grid
        Parameters
        ----------
        iLUT : dict
            calibration LUT from self.calibration_lut
        bounds : dict
            boundaries of block in each swath from self.get_swath_bounds

        Returns
        -------
            noiseLUTgrd : ndarray
                full size noise or calibration matrices for entire image
        '''
        noiseLUTgrd = np.ones((self.numberOfLines, self.numberOfSamples)) * np.nan

        epLen = 100    # extrapolation length
        oLUT = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[], 'pixel':[] }
        for iSW in range(5):
            bound = bounds['EW'+str(iSW+1)]
            xInterp = np.array(range(min(bound['firstRangeSample'])-epLen,
                                     max(bound['lastRangeSample'])+epLen))
            ptsValue = []
            for iVec, iLine in enumerate(iLUT['lines']):
                vecPixel = iLUT['pixels'][iVec]
                vecValue = iLUT['values'][iVec]
                blockIdx = np.nonzero(iLine >= bound['firstAzimuthLine'])[0][-1]
                pix0 = bound['firstRangeSample'][blockIdx]
                pix1 = bound['lastRangeSample'][blockIdx]
                gpi = (vecPixel >= pix0) * (vecPixel <= pix1)
                xPts = vecPixel[gpi]
                yPts = vecValue[gpi]
                interpFtn = InterpolatedUnivariateSpline(xPts, yPts, k=3)
                yInterp = interpFtn(xInterp)
                ptsValue.append(yInterp)

            values = np.stack(ptsValue)
            spline = RectBivariateSpline(iLUT['lines'], xInterp, values, kx=1, ky=1)
            ewLUT = spline(range(iLUT['lines'].min(), iLUT['lines'].max()+1),
                           range(xInterp.min(), xInterp.max()+1))

            for fal, frs, lal, lrs in zip(bound['firstAzimuthLine'],
                                          bound['firstRangeSample'],
                                          bound['lastAzimuthLine'],
                                          bound['lastRangeSample']):
                for iAziLine in range(fal,lal+1):
                    indexShift = xInterp[0]
                    noiseLUTgrd[iAziLine, frs:lrs+1] = ewLUT[iAziLine,
                                                            frs-indexShift:
                                                            lrs-indexShift+1]

        return noiseLUTgrd

    def get_orbit(self, pol):
        ''' Get orbit parameters from XML '''
        orbit = { 'time':[], 'px':[], 'py':[], 'pz':[],
                             'vx':[], 'vy':[], 'vz':[] }
        orbitList = getElem(self.annotXML[pol], ['orbitList'])
        for iOrbit in orbitList.getElementsByTagName('orbit'):
            orbit['time'].append(
                convertTime2Sec(getValue(iOrbit, ['time'])))
            orbit['px'].append(float(getValue(iOrbit, ['position','x'])))
            orbit['py'].append(float(getValue(iOrbit, ['position','y'])))
            orbit['pz'].append(float(getValue(iOrbit, ['position','z'])))
            orbit['vx'].append(float(getValue(iOrbit, ['velocity','x'])))
            orbit['vy'].append(float(getValue(iOrbit, ['velocity','y'])))
            orbit['vz'].append(float(getValue(iOrbit, ['velocity','z'])))

        return orbit

    def __getitem__(self, bandID):
        ''' Apply noise and scaloping gain correction to sigma0_HH/HV '''
        band = self.get_GDALRasterBand(bandID)
        name = band.GetMetadata().get('name', '')
        if name not in ['sigma0_HH', 'sigma0_HV', 'sigma0HH_', 'sigma0HV_']:
            return Nansat.__getitem__(self, bandID)
        if name[-1]=='_':
            pol = name[-3:-1]
        else:
            pol = name[-2:]

        addUpPower = -23.5
        filterOutPower = -25.0

        speedOfLight = 299792458.
        radarFrequency = 5405000454.33435
        azimuthSteeringRate = { 'EW1': 2.390895448 , 'EW2': 2.811502724, \
                                'EW3': 2.366195855 , 'EW4': 2.512694636, \
                                'EW5': 2.122855427                         }

        self.numberOfSamples = int(getValue(self.annotXML[pol], ['numberOfSamples']))
        self.numberOfLines = int(getValue(self.annotXML[pol], ['numberOfLines']))

        orbit = self.get_orbit(pol)

        azimuthFmRate = { 'azimuthTime':[], 't0':[], 'c0':[], 'c1':[], 'c2':[] }
        azimuthFmRateList = getElem(self.annotXML[pol], ['azimuthFmRateList'])
        azimuthFmRates = azimuthFmRateList.getElementsByTagName('azimuthFmRate')
        for iAzimuthFmRate in azimuthFmRates:
            azimuthFmRate['azimuthTime'].append(
                convertTime2Sec(getValue(iAzimuthFmRate, ['azimuthTime'])))
            azimuthFmRate['t0'].append(float(getValue(iAzimuthFmRate,['t0'])))
            tmpValues = getValue(iAzimuthFmRate,
                                 ['azimuthFmRatePolynomial']).split(' ')
            azimuthFmRate['c0'].append(float(tmpValues[0]))
            azimuthFmRate['c1'].append(float(tmpValues[1]))
            azimuthFmRate['c2'].append(float(tmpValues[2]))

        antennaPatternTime = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[] }
        antPatList = getElem(self.annotXML[pol],['antennaPattern','antennaPatternList'])
        for iAntPat in antPatList.getElementsByTagName('antennaPattern'):
            subswathID = getValue(iAntPat, ['swath'])
            antennaPatternTime[subswathID].append(
                convertTime2Sec(getValue(iAntPat, ['azimuthTime'])))

        geolocationGridPoint = { 'azimuthTime':[], 'slantRangeTime':[], \
                                 'line':[], 'pixel':[], 'elevationAngle':[] }
        geoGridPtList = getElem(self.annotXML[pol], ['geolocationGridPointList'])
        geolocationGridPoints = geoGridPtList.getElementsByTagName('geolocationGridPoint')
        for iGeoGridPt in geolocationGridPoints:
            geolocationGridPoint['azimuthTime'].append(
                convertTime2Sec(getValue(iGeoGridPt, ['azimuthTime'])))
            geolocationGridPoint['slantRangeTime'].append(
                            float(getValue(iGeoGridPt, ['slantRangeTime'])))
            geolocationGridPoint['line'].append(
                            float(getValue(iGeoGridPt, ['line'])))
            geolocationGridPoint['pixel'].append(
                            float(getValue(iGeoGridPt, ['pixel'])))
            geolocationGridPoint['elevationAngle'].append( \
                            float(getValue(iGeoGridPt, ['elevationAngle'])))

        subswathCenter = [1504, 3960, 5948, 7922, 9664]    # nominal values
        centerLineIndex = self.numberOfLines / 2
        wavelength = speedOfLight / radarFrequency

        replicaTime = convertTime2Sec(getValue(self.annotXML[pol], ['replicaList',
                                                               'replica',
                                                               'azimuthTime']))
        zdtBias = (replicaTime - antennaPatternTime['EW1'][0]
                  + np.mean(np.diff(antennaPatternTime['EW1'])) / 2)


        bounds = self.get_swath_bounds(pol)

        GRD = {}
        noiseLUT = self.get_calibration_LUT(pol, 'noise')
        GRD['noise'] = self.interpolate_lut(noiseLUT, bounds)

        gridSampleCoord = np.array( (geolocationGridPoint['line'],
                                     geolocationGridPoint['pixel']) ).transpose()
        GRD['pixel'], GRD['line'] = np.meshgrid(range(self.numberOfSamples),
                                                range(self.numberOfLines))
        GRD['azimuthTime'] = griddata(gridSampleCoord,
                                      geolocationGridPoint['azimuthTime'],
                                      (GRD['line'], GRD['pixel']),
                                      method='linear')
        GRD['slantRangeTime'] = griddata(gridSampleCoord,
                                         geolocationGridPoint['slantRangeTime'],
                                         (GRD['line'],GRD['pixel']),
                                         method='linear')
        GRD['elevationAngle'] = griddata(gridSampleCoord,
                                         geolocationGridPoint['elevationAngle'],
                                         (GRD['line'], GRD['pixel']),
                                         method='linear')
        GRD['descallopingGain'] = np.ones((self.numberOfLines,
                                           self.numberOfSamples)) * np.nan

        swathMergeList = getElem(self.annotXML[pol], ['swathMergeList'])
        for iSwathMerge in swathMergeList.getElementsByTagName('swathMerge'):
            subswathID = getValue(iSwathMerge, ['swath'])
            subswathIndex = int(subswathID[-1])-1
            aziAntElemPat = self.azimuthAntennaElementPattern[subswathID]
            aziAntElemAng = self.azimuthAntennaElementPattern['azimuthAngle']

            kw = azimuthSteeringRate[subswathID] * np.pi / 180

            eta = np.copy(GRD['azimuthTime'][:, subswathCenter[subswathIndex]])
            tau = np.copy(GRD['slantRangeTime'][:, subswathCenter[subswathIndex]])
            Vs = np.linalg.norm(
                     np.array([ np.interp(eta,orbit['time'],orbit['vx']),
                                np.interp(eta,orbit['time'],orbit['vy']),
                                np.interp(eta,orbit['time'],orbit['vz']) ]), axis=0)
            ks = 2 * Vs / wavelength * kw
            ka = np.array([ np.interp( eta[loopIdx],
                     azimuthFmRate['azimuthTime'],
                       azimuthFmRate['c0']
                     + azimuthFmRate['c1'] * (tau[loopIdx]-azimuthFmRate['t0'])**1
                     + azimuthFmRate['c2'] * (tau[loopIdx]-azimuthFmRate['t0'])**2 )
                     for loopIdx in range(self.numberOfLines)])
            kt = ka * ks / (ka - ks)
            tw = np.max(np.diff(antennaPatternTime[subswathID])[1:-1])
            zdt = np.array(antennaPatternTime[subswathID]) + zdtBias
            if zdt[0] > eta[0]: zdt = np.hstack([zdt[0]-tw, zdt])
            if zdt[-1] < eta[-1]: zdt = np.hstack([zdt,zdt[-1]+tw])
            for loopIdx in range(len(zdt)):
                idx = np.nonzero(
                          np.logical_and((eta > zdt[loopIdx]-tw/2),
                                         (eta < zdt[loopIdx]+tw/2)))
                eta[idx] -= zdt[loopIdx]
            eta[abs(eta) > tw / 2] = 0
            antsteer = wavelength / 2 / Vs * kt * eta * 180 / np.pi
            ds = np.interp(antsteer, aziAntElemAng, aziAntElemPat)

            swathBoundsList = getElem(iSwathMerge,['swathBoundsList'])
            for iSwathBounds in swathBoundsList.getElementsByTagName('swathBounds'):
                firstAzimuthLine = int(getValue(iSwathBounds,['firstAzimuthLine']))
                firstRangeSample = int(getValue(iSwathBounds,['firstRangeSample']))
                lastAzimuthLine = int(getValue(iSwathBounds,['lastAzimuthLine']))
                lastRangeSample = int(getValue(iSwathBounds,['lastRangeSample']))

                for iAziLine in range(firstAzimuthLine,lastAzimuthLine+1):
                    GRD['descallopingGain'][iAziLine,
                                            firstRangeSample:lastRangeSample+1] = (
                          np.ones(lastRangeSample-firstRangeSample+1)
                        * 10**(ds[iAziLine]/10.))

        sigma0LUT = self.get_calibration_LUT(pol, 'calibration')
        GRD['sigmaNought'] = self.interpolate_lut(sigma0LUT, bounds)

        DN = self['DN_'+pol]
        DN[DN == 0] = np.nan
        if pol=='HH':
            angularDependency = (
                  (0.60717 * np.exp(-0.12296 * GRD['elevationAngle']) + 0.02218) \
                - (0.60717 * np.exp(-0.12296 * 17.0) + 0.02218))
            sigma0 = ((DN**2-GRD['noise']/GRD['descallopingGain'])
                     / GRD['sigmaNought']**2 - angularDependency)
        elif pol=='HV':
            sigma0 = ((DN**2-GRD['noise']/GRD['descallopingGain'])
                     / GRD['sigmaNought']**2 + 10**(addUpPower/10.))

        sigma0[sigma0 < 10**(filterOutPower/10.)] = np.nan

        return 10*np.log10(sigma0)
