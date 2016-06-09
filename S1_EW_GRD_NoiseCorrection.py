import os
import glob
import numpy as np
from xml.dom.minidom import parse
from scipy.interpolate import griddata, InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
from nansat import Nansat
import warnings
warnings.simplefilter("ignore")

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
        LINEAR EQUATION (IN LOG SCALE):
            -0.2447 * ANGLE - 6.6088
    FOR HV CHANNEL,
        THERMAL NOISE SUBTRACTION + SCALOPING CORRECTION

    HOW TO COMPUTE EXACT ZERO DOPPLER TIME, ZDT? THIS IS SOMEWHAT UNCLEAR YET.
    I INTRODUCED zdtBias TO ADJUST IT APPROXIMATELY.
    """

    def __init__(self, fileName, mapperName='', logLevel=30):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__( self, fileName,
                         mapperName=mapperName, logLevel=logLevel)
        self.calibXML = {}
        self.annotXML = {}
        self.auxcalibXML = {}

        for pol in ['HH', 'HV']:
            self.annotXML[pol] = parse(glob.glob(
                '%s/annotation/s1a*-%s-*.xml'
                % (self.fileName,pol.lower()))[0])
            self.calibXML[pol] = {}
            for prod in ['calibration', 'noise']:
                self.calibXML[pol][prod] = parse(glob.glob(
                    '%s/annotation/calibration/%s-*-%s-*.xml'
                    % (self.fileName,prod,pol.lower()))[0])

        manifestXML = parse('%s/manifest.safe' % self.fileName)
        self.IPFver = float( manifestXML.\
                             getElementsByTagName('safe:software')[0].\
                             attributes['version'].value )

        if self.IPFver < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Noise correction cannot be achieved by using this function!\n')
            return
        elif 2.43 <= self.IPFver < 2.60:
            print('\nWARNING: IPF version of input image is lower than 2.60! '
                  'Noise correction result might be wrong!\n')

        try:
            self.auxcalibXML = parse(glob.glob(
                os.path.join( os.path.dirname(os.path.realpath(__file__)),
                              'S1A_AUX_CAL*.SAFE/data/s1a-aux-cal.xml') )[-1])
        except IndexError:
            print('\nERROR: Missing auxiliary product: S1A_AUX_CAL*.SAFE\n\
                   It must be in the same directory with this module.\n\
                   You can get it from https://qc.sentinel1.eo.esa.int/aux_cal')

    def get_AAEP(self, pol):
        ''' Read azimuth antenna elevation pattern from auxiliary XML data
            provided by ESA (https://qc.sentinel1.eo.esa.int/aux_cal)

        Parameters
        ----------
        pol : str
        polarisation: 'HH' or 'HV'

        Returns
        -------
        AAEP : dict
            EW1, EW2, EW3, EW4, EW5, azimuthAngle - 1D vectors
        '''

        keys = ['EW1', 'EW2', 'EW3', 'EW4', 'EW5', 'azimuthAngle']
        AAEP = dict([(key,[]) for key in keys])
        xmldocElem = self.auxcalibXML
        calibParamsList = getElem(xmldocElem,['calibrationParamsList'])
        for iCalibParams in \
            calibParamsList.getElementsByTagName('calibrationParams'):
            subswathID = getValue(iCalibParams,['swath'])
            if subswathID in keys:
                values = []
                polarisation = getValue(iCalibParams,['polarisation'])
                if polarisation==pol:
                    angInc = float(getValue(iCalibParams,
                                 ['azimuthAntennaElementPattern',
                                  'azimuthAngleIncrement']))
                    AAEP[subswathID] = np.array(map(float,getValue(iCalibParams,
                                           ['azimuthAntennaElementPattern',
                                            'values']).split()))
                    numberOfPoints = len(AAEP[subswathID])
        tmpAngle = np.array(range(0,numberOfPoints)) * angInc
        AAEP['azimuthAngle'] = tmpAngle - tmpAngle.mean()

        return AAEP

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
            gli = [   (iLine >= bound['firstAzimuthLine'][0])
                    * (iLine <= bound['lastAzimuthLine'][-1])
                   for iLine in iLUT['lines']                ]
            ptsValue = []
            for iVec, iLine in enumerate(iLUT['lines']):
                vecPixel = np.array(iLUT['pixels'][iVec])
                vecValue = np.array(iLUT['values'][iVec])
                if gli[iVec]:
                    blockIdx = np.nonzero(iLine >= bound['firstAzimuthLine'])[0][-1]
                else:
                    continue
                pix0 = bound['firstRangeSample'][blockIdx]
                pix1 = bound['lastRangeSample'][blockIdx]
                gpi = (vecPixel >= pix0) * (vecPixel <= pix1)
                xPts = vecPixel[gpi]
                yPts = vecValue[gpi]
                interpFtn = InterpolatedUnivariateSpline(xPts, yPts, k=3)
                yInterp = interpFtn(xInterp)
                ptsValue.append(yInterp)

            values = np.vstack(ptsValue)
            spline = RectBivariateSpline( iLUT['lines'][np.nonzero(gli)],
                                          xInterp, values, kx=1, ky=1 )
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

    def get_azimuthFmRate(self, pol):
        ''' Get azimuth FM rate from XML '''
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

        return azimuthFmRate

    def __getitem__(self, bandID):
        ''' Apply noise and scaloping gain correction to sigma0_HH/HV '''
        band = self.get_GDALRasterBand(bandID)
        name = band.GetMetadata().get('name', '')
        if name not in ['sigma0_HH', 'sigma0_HV', 'sigma0HH_', 'sigma0HV_']:
            return Nansat.__getitem__(self, bandID)
        pol = name[-2:]

        IPFver = self.IPFver
        speedOfLight = 299792458.
        radarFrequency = 5405000454.33435
        azimuthSteeringRate = { 'EW1': 2.390895448 , 'EW2': 2.811502724, \
                                'EW3': 2.366195855 , 'EW4': 2.512694636, \
                                'EW5': 2.122855427                         }

        self.numberOfSamples = int(getValue(self.annotXML[pol], ['numberOfSamples']))
        self.numberOfLines = int(getValue(self.annotXML[pol], ['numberOfLines']))

        orbit = self.get_orbit(pol)
        azimuthFmRate = self.get_azimuthFmRate(pol)

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

        wavelength = speedOfLight / radarFrequency

        replicaTime = convertTime2Sec(
                          getValue(self.annotXML[pol],
                                   ['replicaList','replica','azimuthTime'] ))
        zdtBias = (replicaTime - antennaPatternTime['EW1'][0]
                  + np.mean(np.diff(antennaPatternTime['EW1']))/2)

        bounds = self.get_swath_bounds(pol)

        subswathCenter = [
            int(np.mean((   np.array(bounds['EW%d' % idx]['firstRangeSample'])
                          + np.array(bounds['EW%d' % idx]['lastRangeSample']) )/2))
            for idx in (np.arange(5)+1) ]
        interswathBounds = [
            int(np.mean((   np.mean(bounds['EW%d' % idx]['lastRangeSample'])
                          + np.mean(bounds['EW%d' % (idx+1)]['firstRangeSample']) )/2))
            for idx in (np.arange(4)+1) ]

        noiseLUT = self.get_calibration_LUT(pol, 'noise')
        sigma0LUT = self.get_calibration_LUT(pol, 'calibration')
        AAEP = self.get_AAEP(pol)

        GRD = {}
        GRD['noise'] = self.interpolate_lut(noiseLUT, bounds)
        GRD['sigmaNought'] = self.interpolate_lut(sigma0LUT, bounds)

        gridSampleCoord = np.array( (geolocationGridPoint['line'],
                                     geolocationGridPoint['pixel']) ).transpose()
        GRD['pixel'], GRD['line'] = np.meshgrid(range(self.numberOfSamples),
                                                range(self.numberOfLines))
        GRD['azimuthTime'] = griddata(gridSampleCoord,
                                      geolocationGridPoint['azimuthTime'],
                                      (GRD['line'][:,subswathCenter],
                                       GRD['pixel'][:,subswathCenter]),
                                      method='linear')
        GRD['slantRangeTime'] = griddata(gridSampleCoord,
                                         geolocationGridPoint['slantRangeTime'],
                                         (GRD['line'][:,subswathCenter],
                                          GRD['pixel'][:,subswathCenter]),
                                         method='linear')
        GRD['elevationAngle'] = griddata(gridSampleCoord,
                                         geolocationGridPoint['elevationAngle'],
                                         (GRD['line'], GRD['pixel']),
                                         method='linear')
        elevAngle = np.nanmean(GRD['elevationAngle'],axis=0)
        del GRD['pixel'], GRD['line']
        GRD['descallopingGain'] = np.ones((self.numberOfLines,
                                           self.numberOfSamples)) * np.nan
        GRD['subswathIndex'] = np.ones((self.numberOfLines,
                                        self.numberOfSamples)) * np.nan

        GRD['DN'] = self['DN_'+pol]
        GRD['DN'][GRD['DN']==0] = np.nan
        GRD['sigma0'] = np.ones_like(GRD['DN'])*np.nan
        GRD['sigma0'][200:-200,200:-200] = \
            GRD['DN'][200:-200,200:-200]**2 / GRD['sigmaNought'][200:-200,200:-200]**2
        GRD['NEsigma0'] = GRD['noise'] / GRD['sigmaNought']**2
        if 10*np.log10(np.nanmean(GRD['NEsigma0'])) < -40:
            noisePowerPreScalingFactor = 10**(-30.00/10.) / np.nanmean(GRD['NEsigma0'])
        else:
            noisePowerPreScalingFactor = 1.0
        GRD['NEsigma0'] *= noisePowerPreScalingFactor
        noiseScalingCoeff = np.zeros(5)
        sigma0Adjustment = np.zeros(5)
        fitSlopes = np.zeros(5)
        fitIntercepts = np.zeros(5)

        swathMergeList = getElem(self.annotXML[pol], ['swathMergeList'])
        for iSwathMerge in swathMergeList.getElementsByTagName('swathMerge'):
            subswathID = getValue(iSwathMerge, ['swath'])
            subswathIndex = int(subswathID[-1])-1
            aziAntElemPat = AAEP[subswathID]
            aziAntElemAng = AAEP['azimuthAngle']

            kw = azimuthSteeringRate[subswathID] * np.pi / 180
            eta = np.copy(GRD['azimuthTime'][:, subswathIndex])
            tau = np.copy(GRD['slantRangeTime'][:, subswathIndex])
            Vs = np.linalg.norm( np.array(
                    [ np.interp(eta,orbit['time'],orbit['vx']),
                      np.interp(eta,orbit['time'],orbit['vy']),
                      np.interp(eta,orbit['time'],orbit['vz'])  ]), axis=0)
            ks = 2 * Vs / wavelength * kw
            ka = np.array(
                 [ np.interp( eta[loopIdx],
                              azimuthFmRate['azimuthTime'],
                              (   azimuthFmRate['c0']
                                + azimuthFmRate['c1']
                                  * (tau[loopIdx]-azimuthFmRate['t0'])**1
                                + azimuthFmRate['c2']
                                  * (tau[loopIdx]-azimuthFmRate['t0'])**2 ) )
                   for loopIdx in range(self.numberOfLines) ])
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
                    GRD['subswathIndex'][iAziLine,
                                         firstRangeSample:lastRangeSample+1] = (
                          np.ones(lastRangeSample-firstRangeSample+1)
                        * subswathIndex )

        try:
            from noise_scaling_coeff import noise_scaling
        except:
            noiseScalingCoeff = np.array([ 1.0 , 1.0 , 1.0 , 1.0 , 1.0 ])
            sigma0Adjustment = np.array([ 0 , 0 , 0 , 0 , 0 ])
        else:
            if (noisePowerPreScalingFactor != 1.0) and (IPFver < 2.53):
                noiseScalingCoeff = np.array(noise_scaling['<2.53']['noiseScalingCoeff'])
                sigma0Adjustment = np.array(noise_scaling['<2.53']['sigma0Adjustment'])
            elif (noisePowerPreScalingFactor == 1.0) and (2.50 <= IPFver < 2.60):
                noiseScalingCoeff = np.array(noise_scaling['2.50-2.60']['noiseScalingCoeff'])
                sigma0Adjustment = np.array(noise_scaling['2.50-2.60']['sigma0Adjustment'])
            elif (noisePowerPreScalingFactor == 1.0) and (2.60 <= IPFver < 2.70):
                noiseScalingCoeff = np.array(noise_scaling['2.60-2.70']['noiseScalingCoeff'])
                sigma0Adjustment = np.array(noise_scaling['2.60-2.70']['sigma0Adjustment'])
            elif (noisePowerPreScalingFactor == 1.0) and (2.70 <= IPFver < 2.80):
                noiseScalingCoeff = np.array(noise_scaling['2.70-2.80']['noiseScalingCoeff'])
                sigma0Adjustment = np.array(noise_scaling['2.70-2.80']['sigma0Adjustment'])
            else:
                noiseScalingCoeff = np.array([ 1.0 , 1.0 , 1.0 , 1.0 , 1.0 ])
                sigma0Adjustment = np.array([ 0 , 0 , 0 , 0 , 0 ])

        GRD['sigma0'] = GRD['DN']**2 / GRD['sigmaNought']**2
        rawSigma0 = np.nanmean(GRD['sigma0'],axis=0)
        GRD['NEsigma0'] = GRD['noise'] / GRD['sigmaNought']**2 * noisePowerPreScalingFactor
        rawNEsigma0 = np.nanmean(GRD['NEsigma0'],axis=0) / noisePowerPreScalingFactor
        for subswathIndex in range(5):
            subswathMask = (GRD['subswathIndex']==subswathIndex)
            GRD['NEsigma0'][subswathMask] = ( (   GRD['NEsigma0'][subswathMask]
                                         / GRD['descallopingGain'][subswathMask]
                                         * noiseScalingCoeff[subswathIndex] )
                                      - sigma0Adjustment[subswathIndex] )
        del GRD['DN'], GRD['noise'], GRD['sigmaNought'], subswathMask
        meanNoisePower = np.nanmean(GRD['NEsigma0'])
        GRD['NEsigma0'] = GRD['NEsigma0']-meanNoisePower
        calibNEsigma0 = np.nanmean(GRD['NEsigma0'],axis=0)
        sigma0Adjustment = sigma0Adjustment+meanNoisePower

        print('IPF version: %s' % IPFver )
        print('noise pre-scaling coefficient: %s' % noisePowerPreScalingFactor )
        print('noise scaling coefficient: %s' % noiseScalingCoeff )
        print('sigma0 adjustment power: %s' % sigma0Adjustment )

        if pol=='HH':
            angularDependency = (
                  10**(0.2447 * (GRD['elevationAngle']-17.0) /10.) )
            calibS0 = (GRD['sigma0'] - GRD['NEsigma0']) * angularDependency
            del angularDependency
        elif pol=='HV':
            calibS0 = GRD['sigma0'] - GRD['NEsigma0']
            #calibS0 = calibS0+10**(-24.5/10.)

        calibS0[np.nan_to_num(calibS0)<0] = np.nan
        calibSigma0 = np.nanmean(calibS0,axis=0)
        '''
        return 10*np.log10(calibS0), rawSigma0, rawNEsigma0, calibSigma0, \
               calibNEsigma0, elevAngle, noisePowerPreScalingFactor, \
               noiseScalingCoeff, sigma0Adjustment
        '''
        return 10*np.log10(calibS0)
