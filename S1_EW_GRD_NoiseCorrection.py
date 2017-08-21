import matplotlib;    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, warnings
import numpy as np
from xml.dom.minidom import parse, parseString
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.ndimage import convolve
from nansat import Nansat
from nansat.tools import OptionError
from denoising_coeff import get_denoising_coeffs

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


def get_antennaPattern(self, pol):

    antPatList = getElem( self.annotXML[pol],
                          ['antennaPattern','antennaPatternList'] )
    antennaPatterns = []
    for i,iAntPat in enumerate(antPatList.getElementsByTagName('antennaPattern')):
        antennaPattern = { 'swath':[], 'azimuthTime':[], 'slantRangeTime':[],
                           'elevationAngle':[], 'elevationPattern':[],
                           'incidenceAngle':[], 'terrainHeight':[], 'roll':[] }
        antennaPattern['swath'] = getValue(iAntPat,['swath'])
        antennaPattern['azimuthTime'] = convertTime2Sec(
            getValue(iAntPat,['azimuthTime']))
        antennaPattern['slantRangeTime'] = np.array(
            getValue(iAntPat,['slantRangeTime']).split(),dtype=np.float64)
        antennaPattern['elevationAngle'] = np.array(
            getValue(iAntPat,['elevationAngle']).split(),dtype=np.float64)
        antennaPattern['elevationPattern'] = np.array(
            getValue(iAntPat,['elevationPattern']).split(),dtype=np.float64)
        antennaPattern['incidenceAngle'] = np.array(
            getValue(iAntPat,['incidenceAngle']).split(),dtype=np.float64)
        antennaPattern['terrainHeight'] = np.float64(getValue(iAntPat,['terrainHeight']))
        antennaPattern['roll'] = np.float64(getValue(iAntPat,['roll']))
        antennaPatterns.append(antennaPattern)

    return antennaPatterns


    #anxTime = convertTime2Sec(
    #    getElem(self.annotXML[pol],['ascendingNodeTime']).childNodes[0].nodeValue

def rollSteeringLaw(inputAnxTime,inputTimeVec):

    h0 = 707714.8
    h = [8351.5, 8947.0, 23.32, 11.74]
    phi = [3.1495, -1.5655 , -3.1297, 4.7222]
    T_orb = (12*24*60*60)/175.
    w_orb = 2*np.pi/T_orb
    hgt = np.zeros_like(inputTimeVec)
    for i,t in enumerate(inputTimeVec):
        hgt[i] = h0 + sum([ h[j] * np.sin((j+1) * w_orb * (t-inputAnxTime) + phi[j])
                         for j in range(len(h)) ])
    hgt_ref = 711700
    boresight_ref = 29.450
    alpha_roll = 0.0566
    
    return boresight_ref - alpha_roll * (hgt - hgt_ref)/1000.


def getLocalScalingParams(inputSigma0,inputNEsigma0,inputSWindex,windowSize):
    
    nRowsOrig, nColsOrig = inputSigma0.shape
    nRowsProc = (nRowsOrig//windowSize+bool(nRowsOrig%windowSize))*windowSize
    nColsProc = (nColsOrig//windowSize+bool(nColsOrig%windowSize))*windowSize
    sigma0Chunks = np.ones((nRowsProc,nColsProc))*np.nan
    sigma0Chunks[:nRowsOrig,:nColsOrig] = inputSigma0.copy()
    NEsigma0Chunks = np.ones((nRowsProc,nColsProc))*np.nan
    NEsigma0Chunks[:nRowsOrig,:nColsOrig] = inputNEsigma0.copy()
    SWindexChunks = np.zeros((nRowsProc,nColsProc))
    SWindexChunks[:nRowsOrig,:nColsOrig] = inputSWindex.copy()
    del inputSigma0,inputNEsigma0,inputSWindex
    
    sigma0Chunks = [ sigma0Chunks[i*windowSize:(i+1)*windowSize,
                                  j*windowSize:(j+1)*windowSize]
                     for (i,j) in np.ndindex(nRowsProc//windowSize,
                                             nColsProc//windowSize) ]
    NEsigma0Chunks = [ NEsigma0Chunks[i*windowSize:(i+1)*windowSize,
                                      j*windowSize:(j+1)*windowSize]
                       for (i,j) in np.ndindex(nRowsProc//windowSize,
                                               nColsProc//windowSize) ]
    SWindexChunks = [ SWindexChunks[i*windowSize:(i+1)*windowSize,
                                    j*windowSize:(j+1)*windowSize]
                      for (i,j) in np.ndindex(nRowsProc//windowSize,
                                              nColsProc//windowSize) ]
     
    def subfunc_getLocalScalingParams(sigma0Chunk,NEsigma0Chunk,SWindexChunk):
        NESZ = np.ones_like(sigma0Chunk)*np.nan
        SNR = np.ones_like(sigma0Chunk)*np.nan
        extraScaling = np.ones_like(sigma0Chunk)*np.nan
        STD = np.ones_like(sigma0Chunk)*np.nan
        SWindex = np.ones_like(sigma0Chunk)*np.nan
        uniqueIndices = np.unique(SWindexChunk)
        uniqueIndices = uniqueIndices[uniqueIndices>0]  # ignore 0
        for uniqueIndex in uniqueIndices:
            mask = (SWindexChunk==uniqueIndex) * np.isfinite(sigma0Chunk)
            nAll = np.double(mask.sum())
            referencePower = np.sum(sigma0Chunk[mask]-NEsigma0Chunk[mask])
            noisePower = np.sum(NEsigma0Chunk[mask])
            alpha = 1.0
            for j in range(5):
                denoisedPower = sigma0Chunk[mask]-NEsigma0Chunk[mask]*alpha
                nNeg = (denoisedPower <0).sum()
                denoisedPower[denoisedPower <0] = 0
                alpha += ( (np.sum(denoisedPower)-referencePower)
                           / noisePower *(nAll/(nAll-nNeg)) )
            NESZ[mask] = np.mean(NEsigma0Chunk[mask])
            SNR[mask] = np.mean(sigma0Chunk[mask])/np.mean(NEsigma0Chunk[mask])
            extraScaling[mask] = alpha
            STD[mask] = np.nanstd(sigma0Chunk[mask]-NEsigma0Chunk[mask])
            SWindex[mask] = uniqueIndex
        NESZ = np.nanmean(NESZ)
        SNR = np.nanmean(SNR)
        extraScaling = np.nanmean(extraScaling)
        STD = np.nanmean(STD)
        SWindex = np.nanmean(SWindex)
        return NESZ,SNR,extraScaling,STD,SWindex
            
    outputChunks = map(subfunc_getLocalScalingParams,
                       sigma0Chunks,NEsigma0Chunks,SWindexChunks)
    del sigma0Chunks,NEsigma0Chunks,SWindexChunks

    outputNESZ = [ tuple[0] for tuple in outputChunks ]
    outputNESZ = (np.reshape(outputNESZ,
                      [nRowsProc//windowSize,nColsProc//windowSize])
                 )[:nRowsOrig//windowSize,:nColsOrig//windowSize]
    outputSNR = [ tuple[1] for tuple in outputChunks ]
    outputSNR = (np.reshape(outputSNR,
                     [nRowsProc//windowSize,nColsProc//windowSize])
                )[:nRowsOrig//windowSize,:nColsOrig//windowSize]
    outputExtraScaling = [ tuple[2] for tuple in outputChunks ]
    outputExtraScaling = (np.reshape(outputExtraScaling,
                              [nRowsProc//windowSize,nColsProc//windowSize])
                         )[:nRowsOrig//windowSize,:nColsOrig//windowSize]
    outputSTD = [ tuple[3] for tuple in outputChunks ]
    outputSTD = (np.reshape(outputSTD,
                     [nRowsProc//windowSize,nColsProc//windowSize])
                )[:nRowsOrig//windowSize,:nColsOrig//windowSize]
    outputSWindex = [ tuple[4] for tuple in outputChunks ]
    outputSWindex = (np.reshape(outputSWindex,
                     [nRowsProc//windowSize,nColsProc//windowSize])
                    )[:nRowsOrig//windowSize,:nColsOrig//windowSize]

    return outputNESZ,outputSNR,outputExtraScaling,outputSTD,outputSWindex


def adaptiveNoiseScaling(
        inputSigma0,inputNEsigma0,inputSWindex,extraScalingParams,windowSize):

    meanSigma0 = convolve(
        inputSigma0,np.ones((windowSize,windowSize))/windowSize**2.,mode='constant',cval=0.0)
    meanNEsigma0 = convolve(
        inputNEsigma0,np.ones((windowSize,windowSize))/windowSize**2.,mode='constant',cval=0.0)
    meanSWindex = convolve(
        inputSWindex,np.ones((windowSize,windowSize))/windowSize**2.,mode='constant',cval=0.0)
    SNR = 10*np.log10(meanSigma0/meanNEsigma0-1)
    outputNEsigma0 = inputNEsigma0.copy()
    
    for iSW in range(5):
        interpFtn = InterpolatedUnivariateSpline(
                        extraScalingParams['SNR_dB'],
                        extraScalingParams['extraNoiseScalingParams'][iSW], k=3)
        oIdx = np.isfinite(SNR) * (meanSWindex==iSW+1)
        yInterp = interpFtn(SNR[oIdx])
        outputNEsigma0[oIdx] = inputNEsigma0[oIdx] * yInterp
    
    return outputNEsigma0


class Sentinel1Image(Nansat):

    def __init__(self, fileName, mapperName='', logLevel=30):
        ''' Read calibration/annotation XML files and auxiliary XML file '''
        Nansat.__init__( self, fileName,
                         mapperName=mapperName, logLevel=logLevel)
        self.calibXML = {}
        self.annotXML = {}
        self.auxCalibXML = {}
        for pol in ['HH', 'HV']:
            self.annotXML[pol] = parseString(
                                     self.vrt.annotationXMLDict[pol.lower()])
            self.calibXML[pol] = {}
            self.calibXML[pol]['calibration'] = parseString(
                                             self.vrt.calXMLDict[pol.lower()])
            self.calibXML[pol]['noise'] = parseString(
                                             self.vrt.noiseXMLDict[pol.lower()])

        self.nSamples = int(getValue(self.annotXML[pol], ['numberOfSamples']))
        self.nLines = int(getValue(self.annotXML[pol], ['numberOfLines']))

        manifestXML = parseString(self.vrt.manifestXML)
        self.IPFver = float(manifestXML.getElementsByTagName('safe:software')[0]
                                .attributes['version'].value)

        if self.IPFver < 2.43:
            print('\nERROR: IPF version of input image is lower than 2.43! '
                  'Noise correction cannot be achieved using this module. '
                  'Denoising vectors in annotation file are not qualified.\n')
            return
        elif 2.43 <= self.IPFver < 2.53:
            print('\nWARNING: IPF version of input image is lower than 2.53! '
                  'Noise correction result can be wrong.\n')

        try:
            self.auxCalibXML = parse(glob.glob(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'S1A_AUX_CAL*.SAFE/data/s1a-aux-cal.xml') )[-1])
        except IndexError:
            print('\nERROR: Missing auxiliary product: S1A_AUX_CAL*.SAFE\n\
                   It must be in the same directory with this module.\n\
                   You can get it from https://qc.sentinel1.eo.esa.int/aux_cal')


    def get_azimuthAntennaElevationPattern(self, pol):
        ''' Read azimuth antenna elevation pattern from auxiliary XML data
            provided by ESA (https://qc.sentinel1.eo.esa.int/aux_cal)

        Parameters
        ----------
        pol : str
        polarisation: 'HH' or 'HV'

        Returns
        -------
        azimuthAntennaElevationPattern : dict
            EW1, EW2, EW3, EW4, EW5, azimuthAngle - 1D vectors
        '''

        keys = [ 'EW1', 'EW2', 'EW3', 'EW4', 'EW5',
                 'azimuthAngle', 'noiseCalFac' ]
        azimuthAntennaElevationPattern = dict([(key,[]) for key in keys])
        xmldocElem = self.auxCalibXML
        calibParamsList = getElem(xmldocElem,['calibrationParamsList'])
        for iCalibParams in (calibParamsList
                                .getElementsByTagName('calibrationParams')):
            subswathID = getValue(iCalibParams,['swath'])
            if subswathID in keys:
                values = []
                polarisation = getValue(iCalibParams,['polarisation'])
                if polarisation==pol:
                    azimuthAntennaElevationPattern['noiseCalFac'].append(
                        float(getValue(iCalibParams,
                                       ['noiseCalibrationFactor'])))
                    angInc = getValue( iCalibParams,
                                       ['azimuthAntennaElementPattern',
                                        'azimuthAngleIncrement'] )
                    azimuthAntennaElevationPattern[subswathID] = np.array(
                        map(float,getValue( iCalibParams,
                                            ['azimuthAntennaElementPattern',
                                             'values'] ).split()))
                    numberOfPoints = len(
                        azimuthAntennaElevationPattern[subswathID])
        tmpAngle = np.array(range(0,numberOfPoints)) * float(angInc)
        azimuthAntennaElevationPattern['azimuthAngle'] = (
            tmpAngle - tmpAngle.mean() )

        return azimuthAntennaElevationPattern


    def get_antennaPatternTime(self, pol):
    
        antennaPatternTime = { 'EW1':[],'EW2':[],'EW3':[],'EW4':[],'EW5':[] }
        antPatList = getElem( self.annotXML[pol],
                              ['antennaPattern','antennaPatternList'] )
        for iAntPat in antPatList.getElementsByTagName('antennaPattern'):
            subswathID = getValue(iAntPat, ['swath'])
            antennaPatternTime[subswathID].append(
                convertTime2Sec(getValue(iAntPat, ['azimuthTime'])))

        return antennaPatternTime


    def get_calibrationLUT(self, pol, iProd):
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


    def get_swathMergeList(self, pol):
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


    def gen_subswathIndexMap(self, swathBounds, dtype=np.int8):

        subswathIndexMap =  np.zeros((self.nLines,self.nSamples),dtype)
        for iSwathIndex in range(1,6):
            swathBound = swathBounds['EW'+str(iSwathIndex)]
            for fal, frs, lal, lrs in zip(swathBound['firstAzimuthLine'],
                                          swathBound['firstRangeSample'],
                                          swathBound['lastAzimuthLine'],
                                          swathBound['lastRangeSample']):
                subswathIndexMap[fal:lal+1,frs:lrs+1] = iSwathIndex

        return subswathIndexMap


    def interpolate_lut(self, iLUT, bounds, dtype=np.float32):
        ''' Interpolate noise or calibration lut to single full resolution grid
        Parameters
        ----------
        iLUT : dict
            calibration LUT from self.calibration_lut
        bounds : dict
            boundaries of block in each swath from self.get_swathMergeList

        Returns
        -------
            noiseLUTgrd : ndarray
                full size noise or calibration matrices for entire image
        '''
        noiseLUTgrd = np.ones((self.nLines,
                               self.nSamples), dtype) * np.nan

        epLen = 100    # extrapolation length
        oLUT = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[], 'pixel':[] }
        for iSwathIndex in range(1,6):
            bound = bounds['EW'+str(iSwathIndex)]
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
                    blockIdx = np.nonzero(
                                   iLine >= bound['firstAzimuthLine'])[0][-1]
                else:
                    gli[iVec] = False
                    continue
                pix0 = bound['firstRangeSample'][blockIdx]
                pix1 = bound['lastRangeSample'][blockIdx]
                gpi = (vecPixel >= pix0) * (vecPixel <= pix1) * (vecValue > 0)
                if gpi.sum()==0:
                    gli[iVec] = False
                    continue
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


    def get_orbitList(self, pol):
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


    def get_azimuthFmRateList(self, pol):
        ''' Get azimuth FM rate from XML '''
        azimuthFmRate = { 'azimuthTime':[], 't0':[], 'c0':[], 'c1':[], 'c2':[] }
        azimuthFmRateList = getElem(self.annotXML[pol], ['azimuthFmRateList'])
        azimuthFmRates = azimuthFmRateList.getElementsByTagName('azimuthFmRate')
        for iAzimuthFmRate in azimuthFmRates:
            azimuthFmRate['azimuthTime'].append(
                convertTime2Sec(getValue(iAzimuthFmRate, ['azimuthTime'])))
            azimuthFmRate['t0'].append(float(getValue(iAzimuthFmRate,['t0'])))
            if iAzimuthFmRate.getElementsByTagName('azimuthFmRatePolynomial'):
                tmpValues = getValue(iAzimuthFmRate,
                                     ['azimuthFmRatePolynomial']).split(' ')
                azimuthFmRate['c0'].append(float(tmpValues[0]))
                azimuthFmRate['c1'].append(float(tmpValues[1]))
                azimuthFmRate['c2'].append(float(tmpValues[2]))
            else:
                azimuthFmRate['c0'].append(
                                        float(getValue(iAzimuthFmRate,['c0'])))
                azimuthFmRate['c1'].append(
                                        float(getValue(iAzimuthFmRate,['c1'])))
                azimuthFmRate['c2'].append(
                                        float(getValue(iAzimuthFmRate,['c2'])))

        return azimuthFmRate
    
    
    def get_geolocationGridPointList(self, pol):
        ''' Get geolocationGridPoint from XML '''
        geolocationGridPoint = { 'azimuthTime':[], 'slantRangeTime':[],
                                 'line':[], 'pixel':[], 'latitude':[],
                                 'longitude':[], 'height':[],
                                 'incidenceAngle':[], 'elevationAngle':[] }
        geoGridPtList = getElem(self.annotXML[pol],
                                ['geolocationGridPointList'])
        geolocationGridPoints = (
            geoGridPtList.getElementsByTagName('geolocationGridPoint'))
        for iGeoGridPt in geolocationGridPoints:
            geolocationGridPoint['azimuthTime'].append(
                convertTime2Sec(getValue(iGeoGridPt,['azimuthTime'])))
            geolocationGridPoint['slantRangeTime'].append(
                float(getValue(iGeoGridPt,['slantRangeTime'])))
            geolocationGridPoint['line'].append(
                float(getValue(iGeoGridPt,['line'])))
            geolocationGridPoint['pixel'].append(
                float(getValue(iGeoGridPt,['pixel'])))
            geolocationGridPoint['latitude'].append( \
                float(getValue(iGeoGridPt,['latitude'])))
            geolocationGridPoint['longitude'].append( \
                float(getValue(iGeoGridPt,['longitude'])))
            geolocationGridPoint['height'].append( \
                float(getValue(iGeoGridPt,['height'])))
            geolocationGridPoint['incidenceAngle'].append( \
                float(getValue(iGeoGridPt,['incidenceAngle'])))
            geolocationGridPoint['elevationAngle'].append( \
                float(getValue(iGeoGridPt,['elevationAngle'])))

        return geolocationGridPoint
    
    
    def get_focusedBurstLength(self,pol):
        ''' Get full burst length in focused zero-Doppler azimuth time '''
        aziTimeIntevalSLC = 1./float(getValue(self.annotXML[pol],['azimuthFrequency']))
        inputDimsList = getElem(self.annotXML[pol],['inputDimensionsList'])
        inputDims = inputDimsList.getElementsByTagName('inputDimensions')
        nL = np.array([ int(getValue(inputDims[i],['numberOfInputLines'])) for i in range(5) ])
        nB = max([ d for d in range(1,max(nL)/1000) if sum(nL%d)==0 ])
        burstLength = {}
        for i in range(5):
            burstLength['EW%s'%(i+1)] = nL[i]/nB*aziTimeIntevalSLC

        return burstLength

    def get_denoinsing_coefficients(self, pol, expMode='noiseScaling', **kwargs):
        ''' Estimate coefficients from data (experimental mode)
        Params:
            pol : str
                polarisation, HH or HV
            expMode : str
                Experimental mode for computing denoising coefficients. 
                'noiseScaling'
                    noise scaling coefficient estimation
                'powerBalancing'
                    interswath power balancing coefficient estimation
                'localScaling'
                    local SNR dependent extra scaling coefficient estimation
            **kwargs : parameters for self.get_sigma0_GRD()
        '''
        # generate subswath index map
        GRD_SWindex = self.gen_subswathIndexMap(self.get_swathMergeList(pol))
        # get power and nominal noise
        GRD_sigma0, GRD_NEsigma0 = self.get_sigma0_GRD(self, pol, **kwargs)
        
        # set experiment parameters
        bufAzi = 30       # buffer size in azimuth direction
        bufRng = 20       # buffer size in range direction
        nAziSubBlk = 5    # number of blocks in azimuth direction
        subWinSize = 25   # subwindow size to use for local statistics
        
        # compute azimuth subblock boundaries
        aziFullCov = np.where(
                         np.sum(GRD_SWindex!=0,axis=1)==self.nSamples)
        aziMin = aziFullCov[0][0] + bufAzi
        aziMax = aziFullCov[0][-1] - bufAzi
        blkBounds = np.linspace(aziMin,aziMax,
                                nAziSubBlk+1,dtype='uint')
        
        # azimuth subblock-wise processing
        for iBlk in range(nAziSubBlk):
            
            print( 'compute denoising coefficients "%s" ... subblock %d/%d'
                    % (kwargs['expMode'],iBlk+1,nAziSubBlk) )
            
            # azimuth boundaries
            aziMinBlk = int(blkBounds[iBlk])
            aziMaxBlk = int(blkBounds[iBlk+1])
        
            if kwargs['expMode'] in ['noiseScaling','powerBalancing']:
                
                # prepare empty arrays
                noiseScalingEst = np.zeros(nAziSubBlk)
                powerBalancingEst = np.zeros(nAziSubBlk)
                fitSlopes = np.zeros(nAziSubBlk)
                fitOffsets = np.zeros(nAziSubBlk)
                fitResiduals = np.zeros(nAziSubBlk)
                corrCoeffs = np.zeros(nAziSubBlk)
                SWboundsBlk = np.zeros(15)
                
                # subswath-wise processing
                for iSW in range(1,6):
                    
                    # subswath mask
                    mSW = (GRD_SWindex==iSW)
                    
                    # subswath boundaries
                    SWboundsBlk[(iSW-1)*3] = np.mean(
                        [ np.where(mSW[i])[0][0]
                          for i in range(aziMinBlk,aziMaxBlk+1) ])
                    SWboundsBlk[(iSW-1)*3+2] = np.mean(
                        [ np.where(mSW[i])[0][-1]
                          for i in range(aziMinBlk,aziMaxBlk+1) ])
                    SWboundsBlk[(iSW-1)*3+1] = (
                        SWboundsBlk[(iSW-1)*3]+SWboundsBlk[(iSW-1)*3+2] )/2.

                    # range boundaries
                    rngMinBlk = bufRng + max([ np.where(mSW[i])[0][0]
                                    for i in range(aziMinBlk,aziMaxBlk+1) ])
                    rngMaxBlk = -bufRng + min([ np.where(mSW[i])[0][-1]
                                    for i in range(aziMinBlk,aziMaxBlk+1) ])
                                    
                    # subsets
                    sigma0Blk = GRD_sigma0[aziMinBlk:aziMaxBlk+1,
                                           rngMinBlk:rngMaxBlk+1].copy()
                    NEsigma0Blk = GRD_NEsigma0[aziMinBlk:aziMaxBlk+1,
                                               rngMinBlk:rngMaxBlk+1].copy()
                    
                    # discard samples with NaN or lowest 1% power
                    detrended = ( np.nanmean(sigma0Blk-NEsigma0Blk,axis=0)
                                  + np.nanmean(NEsigma0Blk) )
                    lowPow = np.percentile(detrended[detrended > 0],1.)
                    
                    # range sample mask
                    mRng = (detrended > lowPow)
                    
                    # save correlation coefficient
                    corrCoeffs[iSW-1] = np.corrcoef(
                        sigma0Blk.flatten(),NEsigma0Blk.flatten())[0,1]
                                      
                    if kwargs['expMode']=='noiseScaling':
                        sf = np.linspace(0.0,2.0,201)
                    elif kwargs['expMode']=='powerBalancing':
                        # import noise scaling coefficients
                        sf = get_denoising_coeffs(
                                 preScaling,pol,self.IPFver)[0][iSW-1]
                        # duplication for avoiding error in iteration
                        sf = np.array([sf,sf])
                    res = np.zeros_like(sf)                 # residuals
                    # weight factor: use gradient of the noise power
                    wf = np.nanmean(NEsigma0Blk,axis=0)
                    wf = np.abs(np.gradient(wf))
                    wf = (wf-wf.min())/(wf.max()-wf.min())  # normalize
                    #wf = np.ones_like(wf)
                    
                    # iteration to find best-fit (best detrended curve)
                    x = np.arange(rngMinBlk,rngMaxBlk+1)
                    for i,isf in enumerate(sf):
                        y = np.nanmean(sigma0Blk-NEsigma0Blk*isf,axis=0)
                        P = np.polyfit( x[mRng],y[mRng],deg=1,
                                        full=True,w=wf[mRng]  )
                        res[i] = P[1]
                    
                    # find optimal scaling factor and save fit parameters
                    scalingFactor = sf[res==min(res)].mean()
                    y = np.nanmean(sigma0Blk-NEsigma0Blk*scalingFactor,
                                   axis=0)
                    P = np.polyfit( x[mRng],y[mRng],deg=1,full=True )
                    noiseScalingEst[iSW-1] = scalingFactor
                    fitSlopes[iSW-1] = P[0][0]
                    fitOffsets[iSW-1] = P[0][1]
                    fitResiduals[iSW-1] = P[1][0]
                    del sigma0Blk, NEsigma0Blk, mSW

                # compute inter-subswath power balancing coefficients
                interSWboundsBlk = (
                    SWboundsBlk[[2,5,8,11]]+SWboundsBlk[[3,6,9,12]] )/2.
                for i,rngPos in enumerate(interSWboundsBlk):
                    fitPow1 = fitSlopes[i] * rngPos + fitOffsets[i]
                    fitPow2 = fitSlopes[i+1] * rngPos + fitOffsets[i+1]
                    powerBalancingEst[i+1] = fitPow1-fitPow2
                powerBalancingEst = np.cumsum(powerBalancingEst)

                # set reference subswath as EW3 to minimize radiometric bias
                powerBalancingEst -= powerBalancingEst[2]
                
                # print estimated denoising coefficients
                fmt_e3 = {'float_kind':lambda x: "%+.3e" % x}
                print np.array2string(eval(kwargs['expMode']+'Est'),
                                      formatter=fmt_e3)
                print np.array2string(corrCoeffs,formatter=fmt_e3)
                print np.array2string(fitResiduals,formatter=fmt_e3)

                # subsets
                sigma0Blk = GRD_sigma0[aziMinBlk:aziMaxBlk+1,:].copy()
                NEsigma0Blk0 = GRD_NEsigma0[aziMinBlk:aziMaxBlk+1,:].copy()
                SWindexBlk = GRD_SWindex[aziMinBlk:aziMaxBlk+1,:].copy()

                # apply noise scaling, and power balancing
                NEsigma0Blk = NEsigma0Blk0.copy()
                for iSW in range(1,6):
                    mSW = (SWindexBlk==iSW)
                    NEsigma0Blk[mSW] = ( -powerBalancingEst[iSW-1]
                        + NEsigma0Blk0[mSW] * noiseScalingEst[iSW-1] )

                # noise subtraction
                NCsigma0Blk = sigma0Blk - NEsigma0Blk

                # compute mean profiles
                sigma0pfRaw = np.nanmean(sigma0Blk,axis=0)
                sigma0stdRaw = np.nanstd(sigma0Blk,axis=0)
                NEsigma0pfRaw = np.nanmean(NEsigma0Blk0,axis=0)
                NEsigma0pfCor = np.nanmean(NEsigma0Blk,axis=0)
                sigma0pfCor = np.nanmean(NCsigma0Blk,axis=0)
                sigma0stdCor = np.nanstd(NCsigma0Blk,axis=0)
                negPowRate = (100.* np.sum(NCsigma0Blk<0,axis=0)
                                  / np.sum(np.isfinite(NCsigma0Blk),axis=0))
                incAngPf = np.mean(rbsIA(np.arange(aziMinBlk,aziMaxBlk+1),
                                         np.arange(self.nSamples)),axis=0)
                elevAngPf = np.mean(rbsEA(np.arange(aziMinBlk,aziMaxBlk+1),
                                          np.arange(self.nSamples)),axis=0)

                # save estimated parameters
                estParams = { 'IPFver' : self.IPFver,
                              'expMode' : kwargs['expMode'],
                              'noiseScalingEst' : noiseScalingEst,
                              'powerBalancingEst' : powerBalancingEst,
                              'fitResiduals' : fitResiduals,
                              'corrCoeffs' : corrCoeffs,
                              'sigma0pfRaw' : sigma0pfRaw,
                              'NEsigma0pfRaw' : NEsigma0pfRaw,
                              'sigma0pfCor' : sigma0pfCor,
                              'NEsigma0pfCor' : NEsigma0pfCor,
                              'incAngPf' : incAngPf,
                              'elevAngPf' : elevAngPf,
                              'blkBounds' : blkBounds,
                              'negPowRate' : negPowRate,
                              'sigma0stdRaw' : sigma0stdRaw,
                              'sigma0stdCor' : sigma0stdCor            }
                np.savez( self.name[:-5]+'_%d_of_%d_%s.npz'
                              % (iBlk+1,nAziSubBlk,kwargs['expMode']),
                          **estParams )

                # add-up power
                mSW = (SWindexBlk!=0)
                NEsigma0BlkOffset = np.nanmean(NEsigma0Blk[mSW])
                NCsigma0Blk += NEsigma0BlkOffset
                NCsigma0Blk[NCsigma0Blk==0] = np.nan

                # save jpg
                vmin,vmax = np.percentile(
                    sigma0Blk[np.isfinite(sigma0Blk)],(2.5,97.5))
                plt.imsave( self.name[:-5] + '_%d_of_%d_raw.jpg'
                            % (iBlk+1,nAziSubBlk),
                            sigma0Blk, vmin=vmin, vmax=vmax, cmap='gray' )
                plt.imsave( self.name[:-5] + '_%d_of_%d_%s.jpg'
                            % (iBlk+1,nAziSubBlk,kwargs['expMode']),
                            NCsigma0Blk, vmin=vmin, vmax=vmax, cmap='gray' )

                del( mSW, sigma0Blk, NEsigma0Blk0, SWindexBlk, NEsigma0Blk,
                     NCsigma0Blk )

            elif kwargs['expMode']=='localScaling':
                
                # subsets
                bufRng = 30
                sigma0Blk = GRD_sigma0[aziMinBlk:aziMaxBlk+1,
                                       bufRng:-bufRng+1].copy()
                NEsigma0Blk = GRD_NEsigma0[aziMinBlk:aziMaxBlk+1,
                                           bufRng:-bufRng+1].copy()
                SWindexBlk = GRD_SWindex[aziMinBlk:aziMaxBlk+1,
                                         bufRng:-bufRng+1].copy()

                # import coefficients from table
                noiseScaling,powerBalancing = (
                    get_denoising_coeffs(preScaling,pol,self.IPFver)[:2] )

                # apply noise scaling, and power balancing
                meanNEsigma0Blk = np.nanmean(NEsigma0Blk)
                for iSW in range(1,6):
                    mSW = (SWindexBlk==iSW)
                    NEsigma0Blk[mSW] *= noiseScaling[iSW-1]
                    NEsigma0Blk[mSW] -= powerBalancing[iSW-1]

                # total noise power conservation
                NEsigma0Blk -= (np.nanmean(NEsigma0Blk)-meanNEsigma0Blk)
                
                # save correlation coefficient
                corrCoeffs = np.zeros(5)
                for iSW in range(1,6):
                    mSW = (SWindexBlk==iSW) * ((1.1*sigma0Blk-NEsigma0Blk)>0)
                    corrCoeffs[iSW-1] = np.corrcoef(
                        sigma0Blk[mSW],NEsigma0Blk[mSW])[0,1]
                
                # compute local statistics
                localNESZ,localSNR,localExtraScaling,localSTD,localSWindex = (
                    getLocalScalingParams(
                        sigma0Blk,NEsigma0Blk,SWindexBlk,subWinSize) )
                validIdx = ( np.isfinite(localNESZ)
                             * np.isfinite(localSNR)
                             * np.isfinite(localExtraScaling)
                             * np.isfinite(localSTD)
                             * (localSWindex%1==0) )
                
                # save estimated parameters
                estParams = { 'IPFver' : self.IPFver,
                              'expMode' : kwargs['expMode'],
                              'corrCoeffs' : corrCoeffs,
                              'NESZ' : localNESZ[validIdx],
                              'SNR' : localSNR[validIdx],
                              'extraSC' : localExtraScaling[validIdx],
                              'STD' : localSTD[validIdx],
                              'SWindex' : localSWindex[validIdx] }
                
                estParams = {'IPFver' : self.IPFver, 'STD':np.ones(5)*np.nan}
                for iSW in range(1,6):
                    mSW = (SWindexBlk==iSW)
                    estParams['STD'][iSW-1] = np.nanstd(sigma0Blk[mSW])
                np.savez( self.name[:-5]+'_%d_of_%d_%s.npz'
                              % (iBlk+1,nAziSubBlk,kwargs['expMode']),
                          **estParams )

    def get_sigma0_GRD(self, pol, clipDirtyPx=True, bufSize=300, **kwargs):
        ''' Calculate sigma0
        Params:
            pol : str
                polarison, HH or HV
            clipDirtyPx : boolean
                Remove dirty pixels in near-range.
                (see "Masking No-value Pixels on GRD Products generated by the 
                Sentinel-1 ESA IPF", OI-MPC-OTH-0243-S1-Border-Masking-MPCS-916)
            bufSize : int
                Width of left margin where dirty pixels are clipped
         Returns:
            GRD_sigma0 : ndarray
            GRD_NEsigma0 : ndarray
        '''
        # import swathMergeList
        bounds = self.get_swathMergeList(pol)
        
        # generate subswath index map
        GRD_SWindex = self.gen_subswathIndexMap(bounds)
        
        # import LUTs and convert them into full resolution grid
        sigma0LUT = self.get_calibrationLUT(pol,'calibration')
        GRD_radCalCoeff = np.power(self.interpolate_lut(sigma0LUT,bounds),2)
        noiseLUT = self.get_calibrationLUT(pol, 'noise')
        GRD_NEsigma0 = self.interpolate_lut(noiseLUT, bounds) / GRD_radCalCoeff
        
        # import DN and convert into sigma0
        GRD_sigma0 = np.power(self['DN_'+pol].astype('float32'),2) / GRD_radCalCoeff
        GRD_sigma0[GRD_sigma0==0] = np.nan
        del GRD_radCalCoeff
        
        # remove near range dirty pixels
        if clipDirtyPx:
            if pol=='HH':
                mask = np.where(GRD_sigma0[:,:bufSize] < GRD_NEsigma0[:,:bufSize])
            else:
                GRD_sigma0_HH, GRD_NEsigma0_HH = self.get_sigma0_GRD('HH', True, bufSize)
                mask = np.where(GRD_sigma0_HH[:,:bufSize] < GRD_NEsigma0_HH[:,:bufSize])
            
            GRD_sigma0[mask] = np.nan

        return GRD_sigma0, GRD_NEsigma0

    def add_denoised_band(self, bandName='sigma0_HV', **kwargs):
        ''' Remove noise from sigma0 and add array as a band
        Parameters
        ----------
            bandName : str
                Name of the band (e.g. 'sigma0_HH' or 'sigma0_HV')
            **kwargs: parameters for self.get_denoised_band()
        Modifies
        --------
            adds band with name 'sigma0_HH_denoised' to self
        '''
        
        denoisedBandArray = self.get_denoised_band(bandName, **kwargs)
        self.add_band(denoisedBandArray,
                      parameters={'name': bandName + '_denoised'})


    def get_denoised_band(self, bandID, denoAlg='NERSC',
                          addPow=0, adaptNoiSc=True, development=True,
                          fillVoid=False, angDepCor=True, dBconv=True,
                          add_aux_bands=False, **kwargs):
        ''' Apply noise and scaloping gain correction to sigma0_HH/HV
        Params:
            denoAlg : str
                Denoising algorithm to apply. 
                'ESA'
                    Subtract annotated noise vectors from ESA-provided sigma0.
                'NERSC'
                    Subtract scaled and power balanced noise from raw sigma0.
                    Descalloping functionality is also included.
            addPow : float or str
                Add constant power (dB) in order to prevent denoised pixel 
                having negative power. 0 is special case for not adding power, 
                but adjusting the derived noise field have the same total power 
                with ESA-provided noise field. If the input is not a number but 
                one of 'EW['1-5]', the mean noise power of the specific subswath
                will be added to the denoised power. 'EW0' is for whole image.
            adaptNoiSc : boolean
                adaptive noise scaling based on local SNR
            angDepCor : boolean
                Compensate angular dependency from HH-pol image
            fillVoid : boolean
                Fill void pixels (pixels having negative power). 
            dBconv : boolean
                Convert output value to dB scale. 
            add_aux_bands : boolean
                Add auxiliary bands incidence_angle, sigma0_raw, NEsigma0_POL
            development: boolean
                Run variance correction
            **kwargs : dict
                Parameters for self.get_sigma0_GRD()
        Returns:
            GRD_NCsigma0 : ndarray
            Denoised band
        '''
        band = self.get_GDALRasterBand(bandID)
        name = band.GetMetadata().get('name','')
        if name not in ['sigma0_HH', 'sigma0_HV', 'sigma0HH_', 'sigma0HV_']:
            return Nansat.__getitem__(self, bandID)
        pol = name[-2:]

        # import geolocation grid point list from annotation XML
        ggPts = self.get_geolocationGridPointList(pol)
    
        # estimate width and height of geolocation grid
        # normaly width is fixed to 21
        # and height depends on the length of the image product
        ggWidth = np.nonzero(np.diff(ggPts['pixel']) < 0)[0][0] + 1
        ggHeight = len(ggPts['pixel'])/ggWidth
            
        # reshape ggPts to 2D grids
        ggAziT = np.reshape(ggPts['azimuthTime'],(ggHeight, ggWidth))
        ggRngT = np.reshape(ggPts['slantRangeTime'],(ggHeight, ggWidth))
        ggLine = np.reshape(ggPts['line'],(ggHeight,ggWidth))
        ggPixel = np.reshape(ggPts['pixel'],(ggHeight,ggWidth))
        ggIncAng = np.reshape(ggPts['incidenceAngle'],(ggHeight,ggWidth))
        ggElevAng = np.reshape(ggPts['elevationAngle'],(ggHeight,ggWidth))
            
        # train RectBivariateSplines. spline order is fixed to 1
        rbsAT = RectBivariateSpline(
                    ggLine[:,0], ggPixel[0], ggAziT, kx=1, ky=1)
        rbsRT = RectBivariateSpline(
                    ggLine[:,0], ggPixel[0], ggRngT, kx=1, ky=1)
        rbsIA = RectBivariateSpline(
                    ggLine[:,0], ggPixel[0], ggIncAng, kx=1, ky=1)
        rbsEA = RectBivariateSpline(
                    ggLine[:,0], ggPixel[0], ggElevAng, kx=1, ky=1)

        # generate subswath index map
        GRD_SWindex = self.gen_subswathIndexMap(self.get_swathMergeList(pol))
        # get power and nominal thermal noise 
        GRD_sigma0, GRD_NEsigma0 = self.get_sigma0_GRD(pol, **kwargs)

        # denoising using original ESA-provided noise vectors
        if denoAlg=='ESA':
            
            # add-up power
            if addPow in ['EW1','EW2','EW3','EW4','EW5']:
                mSW = (GRD_SWindex==int(addPow[-1]))
                NEsigma0Offset = np.nanmean(GRD_NEsigma0[mSW])
            elif addPow=='EW0':
                mSW = (GRD_SWindex!=0)
                NEsigma0Offset = np.nanmean(GRD_NEsigma0[mSW])
            elif addPow==0:
                NEsigma0Offset = 0
            else:
                NEsigma0Offset = 10**(addPow/10.)
            
            GRD_NCsigma0 = GRD_sigma0 - GRD_NEsigma0 + NEsigma0Offset
        
        # NERSC denoising method
        elif denoAlg=='NERSC':
        
            # set constantant system values
            speedOfLight = 299792458.
            radarFrequency = 5405000454.33435
            wavelength = speedOfLight / radarFrequency
            antennaSteeringRate = { 'EW1': 2.390895448 , 'EW2': 2.811502724,
                                    'EW3': 2.366195855 , 'EW4': 2.512694636,
                                    'EW5': 2.122855427                       }
            
            # import orbit vector list from annotation XML
            orbitVec = self.get_orbitList(pol)
            
            # import azimuth frequency modulation rate list from annotation XML
            aziFmR = self.get_azimuthFmRateList(pol)
            
            # import azimuth time of antenna pattern list from annotation XML
            # these time correspond to burst start time
            antPatT = self.get_antennaPatternTime(pol)
            
            # full burst lenth in focused azimuth time
            burstLength = self.get_focusedBurstLength(pol)
            
            # calculate mean subswath boundaries and centers
            # odd: subswath edges,  even: subswath centers
            meanSWbounds = np.zeros(11)
            for i in range(1,6):
                meanSWbounds[i*2-1] = np.where(GRD_SWindex==i)[1].mean()
                meanSWbounds[i*2] = 2*meanSWbounds[i*2-1]-meanSWbounds[i*2-2]
            meanSWbounds = np.round(meanSWbounds).astype(int).tolist()
            
            # apply RectBivariateSplines to get full resolution grid
            aziTatSWcenter = rbsAT(np.arange(self.nLines),
                                   np.arange(self.nSamples)[meanSWbounds[1::2]])
            rngTatSWcenter = rbsRT(np.arange(self.nLines),
                                   np.arange(self.nSamples)[meanSWbounds[1::2]])


            #####   COMPUTE DESCALLOPING GAIN

            # prepare empty array
            GRD_dsGain = np.ones((self.nLines,self.nSamples),dtype=np.float32)
            # import azimuth antenna elevation pattern from
            # auxiliary calibration XML file, which is available from
            # https://qc.sentinel1.eo.esa.int
            AAEP = self.get_azimuthAntennaElevationPattern(pol)
            
            # compute descalloping gain for each subswath
            for iSW in range(1,6):
                
                # load azimuth antenna element patterns for given subswath
                SWid = 'EW%d' % iSW
                aziAntElemPat = AAEP[SWid]
                aziAntElemAng = AAEP['azimuthAngle']
                
                # antenna steering rate
                kw = antennaSteeringRate[SWid] * np.pi / 180
                
                # azimuth time at subswath center
                eta = np.copy(aziTatSWcenter[:,iSW-1])
                
                # slant range time at subswath center
                tau = np.copy(rngTatSWcenter[:,iSW-1])
                
                # satellite velocity at subswath center
                Vs = np.linalg.norm( np.array(
                        [ np.interp(eta,orbitVec['time'],orbitVec['vx']),
                          np.interp(eta,orbitVec['time'],orbitVec['vy']),
                          np.interp(eta,orbitVec['time'],orbitVec['vz'])  ]),
                                     axis=0 )
                                     
                # Doppler rate introduced by the scanning of the antenna
                ks = 2 * Vs / wavelength * kw
                
                # Doppler rate of point target signal
                ka = np.array( [ np.interp( eta[i], aziFmR['azimuthTime'],
                                  (   aziFmR['c0']
                                    + aziFmR['c1']*(tau[i]-aziFmR['t0'])**1
                                    + aziFmR['c2']*(tau[i]-aziFmR['t0'])**2 ) )
                       for i in range(self.nLines) ])
                       
                # combined Doppler rate (net effect)
                kt = ka * ks / (ka - ks)
                
                # mean burst length in time
                tw = np.max(np.diff(antPatT[SWid])[1:-1])
                
                # zero-Doppler azimuth time at each burst start
                zdt = np.array(antPatT[SWid])
                
                # time difference between
                # full burst in SLC and merged/clipped burst in GRDM
                zdtBias = (burstLength[SWid]-np.diff(zdt))/2
                
                # time correction
                zdt += np.hstack([zdtBias[0], zdtBias])
                
                # if antenna pattern time list does not cover full image,
                # add more sample points using mean burst length tw
                if zdt[0] > eta[0]: zdt = np.hstack([zdt[0]-tw, zdt])
                if zdt[-1] < eta[-1]: zdt = np.hstack([zdt,zdt[-1]+tw])
                
                # wrap azimuth time to burst length
                for i in range(len(zdt)-1):
                    idx = np.nonzero( np.logical_and( (eta >= zdt[i]),
                                                      (eta < zdt[i+1]) ) )
                    eta[idx] -= (zdt[i]+zdt[i+1])/2
                
                # compute antenna steering angle for each burst time
                antSteerAng = wavelength / 2 / Vs * kt * eta * 180 / np.pi
                
                # compute scalloping gain for each burst time
                dsLog = np.interp(antSteerAng, aziAntElemAng, aziAntElemPat)
                
                # convert scale (log to linear)
                dsLin = 10 ** (dsLog / 10.)
                
                # assign computed descalloping gain into each subswath
                mSW = (GRD_SWindex==iSW)
                GRD_dsGain[mSW] = (
                    dsLin[:,np.newaxis]*np.ones(self.nSamples))[mSW]
                del mSW


            #####   APPLY PRE-SCALING AND DE-SCALLOPING
            
            # get pre-scaling factors
            if 10*np.log10(np.nanmean(GRD_NEsigma0)) < -40:
                # denoising vectors not qualified. pre-scaling required.
                preScaling = np.array(AAEP['noiseCalFac'])*(1087**2)
                # scalloping seems to be already applied in given noise vectors
                GRD_dsGain = np.ones(GRD_dsGain.shape)
            else:
                preScaling = [1.0, 1.0, 1.0, 1.0 ,1.0]
            
            # apply pre-scaling
            for iSW in range(1,6):
                mSW = (GRD_SWindex==iSW)
                GRD_NEsigma0[mSW] *= preScaling[iSW-1]
                del mSW
            
            # apply descalloping
            meanNEsigma0 = np.nanmean(GRD_NEsigma0)
            GRD_NEsigma0 /= GRD_dsGain
            del GRD_dsGain
            
            # save unscaled NEsigma0 as a band
            if add_aux_bands:
                self.add_band(GRD_NEsigma0,
                              parameters={'name': 'NEsigma0_'+pol+'_raw'})


            # import coefficients from table (operational mode)
            noiseScaling,powerBalancing,extraScaling = (
                get_denoising_coeffs(preScaling,pol,self.IPFver) )


            #####   APPLY DENOISING COEFFICIENTS AND COMPUTE DENOISED SIGMA0

            # apply noise scaling, and power balancing to NEsigma0
            for iSW in range(1,6):
                mSW = (GRD_SWindex==iSW)
                GRD_NEsigma0[mSW] *= noiseScaling[iSW-1]
                GRD_NEsigma0[mSW] -= powerBalancing[iSW-1]
            self.diffNEsigma0 = (np.nanmean(GRD_NEsigma0)-meanNEsigma0)
            GRD_NEsigma0 -= self.diffNEsigma0

            # local SNR dependent adaptive noise scaling
            if adaptNoiSc and (pol=='HV'):
                GRD_NEsigma0 = adaptiveNoiseScaling(
                    GRD_sigma0,GRD_NEsigma0,GRD_SWindex,extraScaling,5)

            # add-up power
            if addPow in ['EW1','EW2','EW3','EW4','EW5']:
                mSW = (GRD_SWindex==int(addPow[-1]))
                NEsigma0Offset = np.nanmean(GRD_NEsigma0[mSW])
            elif addPow=='EW0':
                mSW = (GRD_SWindex!=0)
                NEsigma0Offset = np.nanmean(GRD_NEsigma0[mSW])
            elif addPow==0:
                NEsigma0Offset = 0
            else:
                NEsigma0Offset = 10**(addPow/10.)

            # noise subtraction
            GRD_NCsigma0 = GRD_sigma0 - GRD_NEsigma0 + NEsigma0Offset
            GRD_NCsigma0[GRD_NCsigma0==0] = np.nan

            # UNDER DEVELOPMENT ...
            if development:
                from sentinel1corrected.underDevelopment import devFtn
                acqDate = os.path.split(self.fileName)[-1][17:25]
                GRD_NCsigma0 = devFtn(
                    self.IPFver,pol,acqDate,GRD_NCsigma0,
                    GRD_NEsigma0,GRD_SWindex,NEsigma0Offset,25)
         
        if fillVoid:
            # find pixels with negative power
            mNeg = (np.nan_to_num(GRD_NCsigma0) < 0)
            # replace them with raw sigma0 (just like SNAP-S1TBX)
            GRD_NCsigma0[mNeg] = GRD_sigma0[mNeg]
            del mNeg

        # angular dependency correction for HH-pol
        if pol=='HH' and angDepCor:
            #angularDependency = 10**( 0.24 * (incidenceAngle-20.0) /10. )
            # estimate incidenceAngle unsing RectBivariateSpline
            incAng = rbsIA(np.arange(self.nLines),
                            np.arange(self.nSamples)).astype(np.float32)
            GRD_angDep = np.power(10, 0.24*(incAng-20.0)/10)
            GRD_NCsigma0 = GRD_NCsigma0 * GRD_angDep
            del incAng,GRD_angDep
                
        if dBconv:
            GRD_sigma0 = 10*np.log10(GRD_sigma0)
            GRD_NEsigma0 = 10*np.log10(GRD_NEsigma0)
            GRD_NCsigma0 = 10*np.log10(GRD_NCsigma0)
                    
        if add_aux_bands:
            # save subswath index, raw sigma0 and scaled NEsigma0 as band
            self.add_band(GRD_SWindex,
                parameters={'name': 'subswath_indices'})
            self.add_band(GRD_sigma0,
                parameters={'name': 'sigma0_'+pol+'_raw'})
            self.add_band(GRD_NEsigma0,
                            parameters={'name': 'NEsigma0_'+pol})

        # return denoised band
        return GRD_NCsigma0
