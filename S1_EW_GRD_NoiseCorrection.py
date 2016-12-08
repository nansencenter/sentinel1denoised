import os, glob, warnings
import numpy as np
from xml.dom.minidom import parse, parseString
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from nansat import Nansat
from nansat.tools import OptionError
from noise_scaling_coeff import noise_scaling

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
        + ANGULAR DEPENDENCY REMOVAL (REFERECE ELEVATION ANGLE = 17.0 DEGREE)
    FOR HV CHANNEL,
        THERMAL NOISE SUBTRACTION + SCALOPING CORRECTION
    """

    def __init__(self, fileName, mapperName='', logLevel=30):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__( self, fileName,
                         mapperName=mapperName, logLevel=logLevel)
        self.calibXML = {}
        self.annotXML = {}
        self.auxcalibXML = {}
        for pol in ['HH', 'HV']:
            self.annotXML[pol] = parseString(self.vrt.annotationXMLDict[pol.lower()])
            self.calibXML[pol] = {}
            self.calibXML[pol]['calibration'] = parseString(
                                            self.vrt.calXMLDict[pol.lower()])
            self.calibXML[pol]['noise'] = parseString(
                                            self.vrt.noiseXMLDict[pol.lower()])

        manifestXML = parseString(self.vrt.manifestXML)
        self.IPFver = float(manifestXML
                                .getElementsByTagName('safe:software')[0]
                                .attributes['version'].value )

        if self.IPFver < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Noise correction cannot be achieved by using this function!\n')
            return
        elif 2.43 <= self.IPFver < 2.53:
            print('\nWARNING: IPF version of input image is lower than 2.53! '
                  'Noise correction result might be wrong!\n')

        try:
            self.auxcalibXML = parse(glob.glob(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
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
        for iCalibParams in (calibParamsList
                                .getElementsByTagName('calibrationParams')):
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

    def interpolate_lut(self, iLUT, bounds, dtype=np.float32):
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
        noiseLUTgrd = np.ones((self.numberOfLines,
                               self.numberOfSamples), dtype) * np.nan

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
            spline = RectBivariateSpline(iLUT['lines'][np.nonzero(gli)],
                                         xInterp, values, kx=1, ky=1 )
            ewLUT = spline(range(iLUT['lines'].min(), iLUT['lines'].max()+1),
                           range(xInterp.min(), xInterp.max()+1)).astype(dtype)

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
    
    def get_geolocGridPts(self, pol):
        ''' Get geolocationGridPoint from XML '''
        geolocationGridPoint = { 'azimuthTime':[], 'slantRangeTime':[],
                                 'line':[], 'pixel':[], 'latitude':[],
                                 'longitude':[], 'elevationAngle':[] }
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
            geolocationGridPoint['latitude'].append( \
                float(getValue(iGeoGridPt, ['latitude'])))
            geolocationGridPoint['longitude'].append( \
                float(getValue(iGeoGridPt, ['longitude'])))

        return geolocationGridPoint

    def add_denoised_band(self, bandName='sigma0_HV', **kwargs):
        ''' Remove noise from sigma0 and add array as a band
        Parameters
        ----------
            bandName : str
                Name of the band (e.g. 'sigma0_HH' or 'sigma0_HV')
            denoising_algorithm : str, optional
                Denoising algorithm to apply. Default is 'NERSC'
                'ESA'
                    Subtract annotated noise vectors from raw sigma0.
                    Pixels having negative values are replaced with raw sigma0.
                    (just like the function 'S-1 Thermal Noise Removal' in ESA SNAP)
                    All other options ('reference_subswath','add_base_power','fill_voids') 
                    are ignored in this case.
                'NERSC'
                    Subtract scaled and power balanced noise from raw sigma0.
                    Descalloping functionality is also included.
            reference_subswath : int, optional
                Reference subswath number for interswath power balancing.
                Valid range is 1 to 5. Default is 1.
            add_base_power : float, optional
                Add constant power (dB) in order to prevent denoised pixel having negative power.
                0 for special case of zero noise power in the reference subswath.
                Default is 0.
            fill_voids : boolean, optional
                Fill void pixels (pixels having negative power). Default is 'False'
                True
                    Replace with its absolute value.
                False
                    Replace with NaN.
            dB_conversion : boolean, optional
                Convert output value to dB scale. Default is 'True'
                True
                    Output sigma0 is in dB scale.
                False
                    Output sigma0 is in linear scale.
        Modifies
        --------
            adds band with name 'sigma0_HH_denoised' to self
        '''
        
        for key in kwargs:
            if key not in [ 'denoising_algorithm' , 'reference_subswath' ,
                            'add_base_power' , 'fill_voids' , 'dB_conversion']:
                raise KeyError("add_denoised_band() got an unexpected keyword argument '%s'" % key)
        if 'denoising_algorithm' not in kwargs:
            kwargs['denoising_algorithm'] = 'NERSC'
        elif kwargs['denoising_algorithm'] not in ['ESA','NERSC']:
            raise KeyError("kwargs['denoising_algorithm'] got an invalid value '%s'"
                           % kwargs['denoising_algorithm'])
        if 'reference_subswath' not in kwargs:
            kwargs['reference_subswath'] = 1
        elif np.int(kwargs['reference_subswath']) not in [1,2,3,4,5]:
            raise KeyError("kwargs['reference_subswath'] got an invalid value '%s'"
                           % kwargs['reference_subswath'])
        if 'add_base_power' not in kwargs:
            kwargs['add_base_power'] = 0
        elif not np.isreal(np.float(kwargs['add_base_power'])):
            raise KeyError("kwargs['add_base_power'] got an invalid value '%s'"
                           % kwargs['add_base_power'])
        if 'fill_voids' not in kwargs:
            kwargs['fill_voids'] = False
        elif not isinstance(kwargs['fill_voids'],bool):
            raise KeyError("kwargs['fill_voids'] got an invalid value '%s'"
                           % kwargs['fill_voids'])
        if 'dB_conversion' not in kwargs:
            kwargs['dB_conversion'] = True
        elif not isinstance(kwargs['dB_conversion'],bool):
            raise KeyError("kwargs['dB_conversion'] got an invalid value '%s'"
                           % kwargs['dB_conversion'])

        denoisedBandArray = self.get_denoised_band(bandName, **kwargs)
        self.add_band(denoisedBandArray,
                      parameters={'name': bandName + '_denoised'})

    def get_denoised_band(self, bandID, **kwargs):
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

        geolocationGridPoint = self.get_geolocGridPts(pol)

        wavelength = speedOfLight / radarFrequency
        '''
        replicaTime = convertTime2Sec(
                          getValue(self.annotXML[pol],
                                   ['replicaList','replica','azimuthTime'] ))
        zdtBias = (replicaTime - antennaPatternTime['EW1'][0]
                  + np.mean(np.diff(antennaPatternTime['EW1']))/2)
        '''
        zeroDopMinusAcqTime = float(getValue(self.annotXML[pol],['zeroDopMinusAcqTime']))
        zdtBias = np.remainder( zeroDopMinusAcqTime ,
                               np.median(np.diff(antennaPatternTime['EW1'])) )
        zdtBias = zdtBias + 0.07

        bounds = self.get_swath_bounds(pol)

        subswathCenter = [
            int(np.mean((   np.array(bounds['EW%d' % idx]['firstRangeSample'])
                          + np.array(bounds['EW%d' % idx]['lastRangeSample']) )/2))
            for idx in (np.arange(5)+1) ]
        interswathBounds = [
            int(np.mean((   np.mean(bounds['EW%d' % idx]['lastRangeSample'])
                          + np.mean(bounds['EW%d' % (idx+1)]['firstRangeSample']) )/2))
            for idx in (np.arange(4)+1) ]


        ## get GRD_elevationAngle
        # estimate width and height of geolocation grid
        ggWidth = np.nonzero(np.diff(geolocationGridPoint['pixel']) < 0)[0][0] + 1
        ggHeight = (len(geolocationGridPoint['pixel']) / ggWidth)
        # reshape geolocationGridPoint to 2D grids
        ggPixels = np.reshape(geolocationGridPoint['pixel'], (ggHeight, ggWidth))
        ggLines = np.reshape(geolocationGridPoint['line'], (ggHeight, ggWidth))
        ggEvationAngles = np.reshape(geolocationGridPoint['elevationAngle'],
                                     (ggHeight, ggWidth))
        ggAzimuthTimes = np.reshape(geolocationGridPoint['azimuthTime'],
                                     (ggHeight, ggWidth))
        ggSlantRangeTimes = np.reshape(geolocationGridPoint['slantRangeTime'],
                                     (ggHeight, ggWidth))
        # train RectBivariateSplines
        rbsEA = RectBivariateSpline(ggLines[:,0], ggPixels[0], ggEvationAngles, kx=1, ky=1)
        rbsAT = RectBivariateSpline(ggLines[:,0], ggPixels[0], ggAzimuthTimes, kx=1, ky=1)
        rbsSRT = RectBivariateSpline(ggLines[:,0], ggPixels[0], ggSlantRangeTimes, kx=1, ky=1)

        # apply RectBivariateSplines to estimate azimuthTime and slantRangeTime
        lines_fullres = np.arange(self.numberOfLines)
        pixels_fullres = np.arange(self.numberOfSamples)
        azimuthTimeAtSubswathCenter = rbsAT(lines_fullres, pixels_fullres[subswathCenter])
        slantRangeTimeAtSubswathCenter = rbsSRT(lines_fullres, pixels_fullres[subswathCenter])

        ## estimate GRD_descallopingGain
        GRD_descallopingGain = np.ones((self.numberOfLines,
                                        self.numberOfSamples),dtype=np.float32) * np.nan
        GRD_subswathIndex = np.ones((self.numberOfLines,
                                     self.numberOfSamples),dtype=np.int8) * (-1)
        AAEP = self.get_AAEP(pol)
        swathMergeList = getElem(self.annotXML[pol], ['swathMergeList'])
        for iSwathMerge in swathMergeList.getElementsByTagName('swathMerge'):
            subswathID = getValue(iSwathMerge, ['swath'])
            subswathIndex = int(subswathID[-1])-1
            aziAntElemPat = AAEP[subswathID]
            aziAntElemAng = AAEP['azimuthAngle']

            kw = azimuthSteeringRate[subswathID] * np.pi / 180
            eta = np.copy(azimuthTimeAtSubswathCenter[:, subswathIndex])
            tau = np.copy(slantRangeTimeAtSubswathCenter[:, subswathIndex])
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
            ds_scaled = 10 ** (ds / 10.)

            swathBoundsList = getElem(iSwathMerge,['swathBoundsList'])
            for iSwathBounds in swathBoundsList.getElementsByTagName('swathBounds'):
                firstAzimuthLine = int(getValue(iSwathBounds,['firstAzimuthLine']))
                firstRangeSample = int(getValue(iSwathBounds,['firstRangeSample']))
                lastAzimuthLine = int(getValue(iSwathBounds,['lastAzimuthLine']))
                lastRangeSample = int(getValue(iSwathBounds,['lastRangeSample']))
                
                for iAziLine in range(firstAzimuthLine,lastAzimuthLine+1):
                    GRD_descallopingGain[iAziLine,
                      firstRangeSample:lastRangeSample+1] = ds_scaled[iAziLine]
                    GRD_subswathIndex[iAziLine,
                            firstRangeSample:lastRangeSample+1] = subswathIndex
                # for loop can be replaced by following matrix multiplication.
                '''
                GRD_descallopingGain[firstAzimuthLine:lastAzimuthLine+1,
                                     firstRangeSample:lastRangeSample+1] = (
                    ds_scaled[firstAzimuthLine:lastAzimuthLine+1][:,np.newaxis]
                    * np.ones((1,lastRangeSample-firstRangeSample+1)) )
                GRD_subswathIndex[firstAzimuthLine:lastAzimuthLine+1,
                    firstRangeSample:lastRangeSample+1] = subswathIndex
                '''

        # estimate noisePowerPreScalingFactor and GRD_NEsigma0
        noiseLUT = self.get_calibration_LUT(pol, 'noise')
        sigma0LUT = self.get_calibration_LUT(pol, 'calibration')

        GRD_radCalCoeff2 = self.interpolate_lut(sigma0LUT, bounds)
        GRD_radCalCoeff2 = np.power(GRD_radCalCoeff2, 2, GRD_radCalCoeff2)

        # sigma0 = DN ** 2 / radCalCoeff ** 2
        GRD_sigma0 = self['DN_'+pol].astype('float32')
        GRD_sigma0 = np.power(GRD_sigma0, 2, GRD_sigma0)
        GRD_sigma0[GRD_sigma0==0] = np.nan
        GRD_sigma0 /= GRD_radCalCoeff2
        rawSigma0 = np.nanmedian(GRD_sigma0,axis=0)

        # NEsigma0 = noiseLUT / radCalCoeff**2 / descallopingGain
        GRD_NEsigma0 = self.interpolate_lut(noiseLUT, bounds)
        GRD_NEsigma0 /= GRD_radCalCoeff2
        del GRD_radCalCoeff2
        
        NEsigma0Mean = np.nanmean(GRD_NEsigma0)
        if 10*np.log10(NEsigma0Mean) < -40:
            noisePowerPreScalingFactor = 10**(-30.00/10.) / NEsigma0Mean
        else:
            noisePowerPreScalingFactor = 1.0

        #runMode = 'HVnoiseScaling'
        #runMode = 'HVbalancingPower'
        #runMode = 'HHbalancingPower'
        runMode = 'operational'
        numberOfAzimuthSubBlock = 5
        noiseAdjRefSW = int(kwargs['reference_subswath']) - 1

        if runMode == 'operational':
            # load IPFVer specific coefficients
            numberOfAzimuthSubBlock = 1
            noiseScalingCoeff,balancingPower = (
                noise_scaling(noisePowerPreScalingFactor,pol,IPFver) )
            NEsigma0SW3 = np.nanmean(GRD_NEsigma0[GRD_subswathIndex==2])
            balancingPower = np.array(balancingPower)
            balancingPowerRef = balancingPower[noiseAdjRefSW]
            if np.float(kwargs['add_base_power'])==0:
                self.NEsigma0Offset = (
                    np.nanmean(GRD_NEsigma0[GRD_subswathIndex==noiseAdjRefSW]))
            else:
                self.NEsigma0Offset = 10**(np.float(kwargs['add_base_power'])/10.)
            GRD_NEsigma0 *= noisePowerPreScalingFactor
            GRD_NEsigma0 /= GRD_descallopingGain
            del GRD_descallopingGain
        else:
            sideCutN = 15

            noiseScalingCoeff = np.zeros((5,numberOfAzimuthSubBlock))
            fitSlopes = np.zeros((5,numberOfAzimuthSubBlock))
            fitIntercepts = np.zeros((5,numberOfAzimuthSubBlock))
            fitResiduals = np.zeros((5,numberOfAzimuthSubBlock))
            meanRawSigma0 = np.zeros((5,numberOfAzimuthSubBlock))
            minAzimuthIndex = max([min(bounds['EW'+str(i+1)]['firstAzimuthLine']) for i in range(5)])
            maxAzimuthIndex = min([max(bounds['EW'+str(i+1)]['lastAzimuthLine']) for i in range(5)])
            subBlockStartIndex = np.linspace(minAzimuthIndex,maxAzimuthIndex,
                                             numberOfAzimuthSubBlock+1,dtype='uint')[:-1]
            subBlockEndIndex = np.linspace(minAzimuthIndex,maxAzimuthIndex,
                                           numberOfAzimuthSubBlock+1,dtype='uint')[1:]
            subBlockCenterIndex = (subBlockStartIndex+subBlockEndIndex)/2

            for iSubswathIndex in range(5):
                minRangeIndex = max(bounds['EW'+str(iSubswathIndex+1)]['firstRangeSample'])
                maxRangeIndex = min(bounds['EW'+str(iSubswathIndex+1)]['lastRangeSample'])
                subswathMask = (GRD_subswathIndex==iSubswathIndex)
                validRangeMask = np.logical_and(
                    np.logical_and( np.arange(self.numberOfSamples)>=(minRangeIndex+sideCutN),
                                    np.arange(self.numberOfSamples)<=(maxRangeIndex-sideCutN) ),
                    10*np.log10(np.nanmin(GRD_sigma0,axis=0))>=-40.)
                if sum(validRangeMask) < 0.5*(maxRangeIndex-minRangeIndex):
                    validRangeMask = np.logical_and(
                        np.arange(self.numberOfSamples)>=(minRangeIndex+sideCutN),
                        np.arange(self.numberOfSamples)<=(maxRangeIndex-sideCutN) )

                for iSubBlockIndex in range(numberOfAzimuthSubBlock):
                    sigma0SW = np.nanmedian( (GRD_sigma0*subswathMask)
                                             [subBlockStartIndex[iSubBlockIndex]:
                                              subBlockEndIndex[iSubBlockIndex]+1],axis=0 )
                    meanRawSigma0[iSubswathIndex,iSubBlockIndex] = np.nanmean(sigma0SW[validRangeMask])
                    NEsigma0SW = np.nanmedian( (GRD_NEsigma0/GRD_descallopingGain*subswathMask)
                                               [subBlockStartIndex[iSubBlockIndex]:
                                                subBlockEndIndex[iSubBlockIndex]+1],axis=0 )
                    ###NEsigma0SW = NEsigma0SW-np.nanmean(NEsigma0SW[validRangeMask])
                    NEsigma0SW = NEsigma0SW-NEsigma0SW[np.nonzero(NEsigma0SW)].mean()
                    rangeIndex = np.arange(sigma0SW.shape[0])[validRangeMask]
                    # Consider adopting Newton method for efficient computation
                    if runMode == 'HVnoiseScaling':
                        scalingFactor = np.linspace(0.0,2.0,201)
                    elif np.logical_or(runMode=='HHbalancingPower',runMode=='HVbalancingPower'):
                        scalingFactor = np.array([
                            noise_scaling(noisePowerPreScalingFactor,pol,IPFver)[0][iSubswathIndex],
                            noise_scaling(noisePowerPreScalingFactor,pol,IPFver)[0][iSubswathIndex] ])
                    slopes = np.zeros_like(scalingFactor)
                    intercepts = np.zeros_like(scalingFactor)
                    residuals = np.zeros_like(scalingFactor)
                    weightFactor = NEsigma0SW[rangeIndex]
                    weightFactor = ((  (weightFactor-np.min(weightFactor))
                                     /(np.max(weightFactor)-np.min(weightFactor)))+1)/2.
                    #weightFactor = np.ones_like(weightFactor)

                    for i,sf in enumerate(scalingFactor):
                        denoisedPower = (sigma0SW-NEsigma0SW*sf)[rangeIndex]
                        if sum(denoisedPower<0) >= len(denoisedPower)*0.3:
                            residuals[i] = +np.inf
                            continue
                        else:
                            denoisedPower = 10*np.log10(denoisedPower)
                            vM = np.isfinite(denoisedPower)
                            P = np.polyfit( rangeIndex[vM],denoisedPower[vM],deg=1,
                                            full='True',w=weightFactor[vM])
                            slopes[i], intercepts[i] = P[0]
                            residuals[i] = P[1]

                    bestFitIndex = np.where(residuals==min(residuals))[0][0]
                    noiseScalingCoeff[iSubswathIndex,iSubBlockIndex] = scalingFactor[bestFitIndex]
                    fitSlopes[iSubswathIndex,iSubBlockIndex] = slopes[bestFitIndex]
                    fitIntercepts[iSubswathIndex,iSubBlockIndex] = intercepts[bestFitIndex]
                    fitResiduals[iSubswathIndex,iSubBlockIndex] = residuals[bestFitIndex]


                    sigma0SW = np.nanmean( (GRD_sigma0*subswathMask)
                                            [subBlockStartIndex[iSubBlockIndex]:
                                             subBlockEndIndex[iSubBlockIndex]+1],axis=0 )
                    NEsigma0SW = np.nanmean( (GRD_NEsigma0/GRD_descallopingGain*subswathMask)
                                              [subBlockStartIndex[iSubBlockIndex]:
                                               subBlockEndIndex[iSubBlockIndex]+1],axis=0 )
                    NEsigma0SW = NEsigma0SW-NEsigma0SW[np.nonzero(NEsigma0SW)].mean()
                    denoisedPower = 10*np.log10((sigma0SW-NEsigma0SW*scalingFactor[bestFitIndex])[rangeIndex])
                    vM = np.isfinite(denoisedPower)
                    P = np.polyfit( rangeIndex[vM],denoisedPower[vM],deg=1,
                                    full='True',w=weightFactor[vM])
                    fitSlopes[iSubswathIndex,iSubBlockIndex],fitIntercepts[iSubswathIndex,iSubBlockIndex] = P[0]
                    fitResiduals[iSubswathIndex,iSubBlockIndex] = P[1]

            balancingPower = np.zeros_like(noiseScalingCoeff)
            boundsPower = np.zeros((13,numberOfAzimuthSubBlock))
            for i,iswc in enumerate(subswathCenter):
                for isb in range(numberOfAzimuthSubBlock):
                    boundsPower[3*i][isb] = 10**((fitSlopes[i][isb] * iswc + fitIntercepts[i][isb])/10.)
            for i,iswb in enumerate(interswathBounds):
                for isb in range(numberOfAzimuthSubBlock):
                    boundsPower[3*i+1][isb] = 10**((fitSlopes[i][isb] * iswb + fitIntercepts[i][isb])/10.)
                    boundsPower[3*i+2][isb] = 10**((fitSlopes[i+1][isb] * iswb + fitIntercepts[i+1][isb])/10.)
                    balancingPower[i+1][isb] = boundsPower[3*i+1][isb]-boundsPower[3*i+2][isb]
            for isb in range(numberOfAzimuthSubBlock):
                balancingPower[:,isb] = np.cumsum(balancingPower[:,isb])
                balancingPower[:,isb] -= balancingPower[2,isb]

            GRD_NEsigma0 *= noisePowerPreScalingFactor
            GRD_NEsigma0 /= GRD_descallopingGain
            del GRD_descallopingGain

        noiseScalingFit = np.zeros((2,5))
        for iSubswathIndex in range(5):
            if numberOfAzimuthSubBlock==1:
                noiseScalingFit[:,iSubswathIndex] = [0,noiseScalingCoeff[iSubswathIndex]]
            else:
                '''
                P = np.polyfit( subBlockCenterIndex,
                                noiseScalingCoeff[iSubswathIndex,:],
                                w=1/fitResiduals[iSubswathIndex,:],
                                deg=1,full='True' )
                noiseScalingFit[:,iSubswathIndex] = P[0]
                '''
                noiseScalingFit[:,iSubswathIndex] = [0,np.median(noiseScalingCoeff[iSubswathIndex])]

        balancingPowerFit = np.zeros((2,5))
        for iSubswathIndex in range(5):
            if numberOfAzimuthSubBlock==1:
                balancingPowerFit[:,iSubswathIndex] = [0,balancingPower[iSubswathIndex]]
            else:
                '''
                P = np.polyfit( subBlockCenterIndex,
                                balancingPower[iSubswathIndex,:],
                                deg=1,full='True' )
                balancingPowerFit[:,iSubswathIndex] = P[0]
                '''
                balancingPowerFit[:,iSubswathIndex] = [0,np.median(balancingPower[iSubswathIndex])]

        rawNEsigma0 = np.nanmedian(GRD_NEsigma0,axis=0) / noisePowerPreScalingFactor

        # FOR HH, USE ESA PROVIDED NOISE VECTOR FOR NOW. APPLY DESCALLOPING.
        if pol == 'HH':
            balancingPowerFit = np.zeros((2,5))
        # FOR HV, USE SCALED AND POWER BALANCED NOISE FIELD. APPLY DESCALLOPING.
        elif pol == 'HV':
            for iSubswathIndex in range(5):
                GRD_NEsigma0[GRD_subswathIndex==iSubswathIndex] -= (
                    np.nanmean(GRD_NEsigma0[GRD_subswathIndex==iSubswathIndex]) )
                noiseScalingModel = ( noiseScalingFit[0,iSubswathIndex]
                                     * np.arange(self.numberOfLines)
                                     + noiseScalingFit[1,iSubswathIndex] )
                for iAzimuthLine in range(self.numberOfLines):
                    subswathMask = (GRD_subswathIndex[iAzimuthLine,:]==iSubswathIndex)
                    GRD_NEsigma0[iAzimuthLine,subswathMask] *= (
                        noiseScalingModel[iAzimuthLine] )
            # CAUTION! IF MEAN NOISE SUBTRACTION MUST BE DONE USING SUBSWATH MEAN.
            # IF THE SUBTRACTION IS DONE IN EACH AZIMUTH LINE WISE,
            # THEN DESCALLOPING DOES NOT WORK
            GRD_NEsigma0 = GRD_NEsigma0 + NEsigma0SW3
        
        if kwargs['denoising_algorithm'] == 'ESA':
            # IN ESA METHOD, NEGATIVE VALUES ARE REPLACED WITH RAW SIGMA0 VALUES
            GRD_NEsigma0 = ( self.interpolate_lut(noiseLUT, bounds) /
                             np.power( self.interpolate_lut(sigma0LUT, bounds), 2) )
            GRD_NCsigma0 = GRD_sigma0 - GRD_NEsigma0
            negativeIdx = np.nan_to_num(GRD_NCsigma0) < 0
            GRD_NCsigma0[negativeIdx] = GRD_sigma0[negativeIdx]
        elif kwargs['denoising_algorithm'] == 'NERSC':
            # IN NERSC METHOD, NEGATIVE VALUES ARE REPLACED WITH ABSOLUTE VALUES
            if runMode != 'HHbalancingPower':
                for iSubswathIndex in range(5):
                    balancingPowerModel = ( balancingPowerFit[0,iSubswathIndex]
                                            * np.arange(self.numberOfLines)
                                            + balancingPowerFit[1,iSubswathIndex] )
                    for iAzimuthLine in range(self.numberOfLines):
                        subswathMask = (GRD_subswathIndex[iAzimuthLine,:]==iSubswathIndex)
                        GRD_NEsigma0[iAzimuthLine,subswathMask] -= (
                            balancingPowerModel[iAzimuthLine] )
                GRD_NCsigma0 = GRD_sigma0 - GRD_NEsigma0 + self.NEsigma0Offset
            if runMode == 'operational' and kwargs['fill_voids']:
                GRD_NCsigma0 = np.abs(GRD_NCsigma0)

        calNEsigma0 = np.nanmedian(GRD_NEsigma0,axis=0)

        if pol == 'HH' and kwargs['denoising_algorithm'] != 'ESA':
            #angularDependency = 10**(0.271 * (elevationAngle-17.0) /10.) )
            # estimate elevationAngle unsing RectBivariateSpline
            GRD_angularDependency = rbsEA(lines_fullres,
                                          pixels_fullres).astype(np.float32)
            elevAngle = np.nanmean(GRD_angularDependency, axis=0)
            GRD_angularDependency -= 17.0
            GRD_angularDependency /= ( 10. / 0.271)
            GRD_angularDependency = np.power(10, GRD_angularDependency,
                                                 GRD_angularDependency)
            GRD_NCsigma0 = GRD_NCsigma0 * GRD_angularDependency
            del GRD_angularDependency

        GRD_NCsigma0[np.isnan(GRD_sigma0)] = np.nan
        calSigma0 = np.nanmedian(GRD_NCsigma0,axis=0)
        '''
        return GRD_sigma0, GRD_NCsigma0, rawSigma0, rawNEsigma0, calSigma0, \
               calNEsigma0, elevAngle, noisePowerPreScalingFactor, \
               noiseScalingCoeff, balancingPower, meanRawSigma0, \
               noiseScalingFit, balancingPowerFit, boundsPower
        '''
        if kwargs['dB_conversion']:
            GRD_NEsigma0 = 10*np.log10(GRD_NEsigma0)
            GRD_NCsigma0 = 10*np.log10(GRD_NCsigma0)

        self.add_band(GRD_NEsigma0,parameters={'name': 'NEsigma0_' + pol})

        return GRD_NCsigma0
