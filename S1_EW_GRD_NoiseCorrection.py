import os
import glob
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString

import numpy as np
from scipy.interpolate import griddata, InterpolatedUnivariateSpline

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

def getAnnoLut(iNansat,iPol,iProd):
    if not (iProd=='calibration' or iProd=='noise'):
        raise ValueError('iProd must be calibration or noise')
    fileIndexDict = { 'HH':u'001', 'HV':u'002' }
    productDict = { 'calibration':'sigmaNought', 'noise':'noiseLut' }
    xmldocElem = parse( glob.glob(\
        iNansat.fileName + '/annotation/calibration/' + iProd \
        + '*' + fileIndexDict[iPol] + '.xml' )[0] )
    oLUT = { 'pixel':[], 'line':[], 'value':[] }
    vectorList = getElem(xmldocElem,[iProd+'VectorList'])
    for iVector in vectorList.getElementsByTagName(iProd+'Vector'):
        oLUT['pixel'].append(
            map(int, getValue(iVector,['pixel']).split()))
        oLUT['line'].append(int(getValue(iVector,['line'])))
        oLUT['value'].append(
            map(float, getValue(iVector,[productDict[iProd]]).split()))
    return oLUT


def restoreNoiseLUT(iLUT):
    swPts = [0, 3020, 4930, 7000, 8870, 10400]    # nominal subswath boundaries
    epLen = 100    # extrapolation length
    ptsPixel = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[] }
    ptsLine = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[] }
    ptsValue = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[] }
    oLUT = { 'EW1':[], 'EW2':[], 'EW3':[], 'EW4':[], 'EW5':[], 'pixel':[] }
    for iVec,iLine in enumerate(iLUT['line']):
        vecPixel = iLUT['pixel'][iVec]
        vecValue = iLUT['value'][iVec]
        startIndex = np.insert(np.where(np.diff(vecPixel)==1)[0]+1,0,0)
        startIndex = startIndex[np.append(np.diff(startIndex)!=1,True)]
        endIndex = np.append(np.where(np.diff(vecPixel)==1)[0],len(vecPixel)-1)
        endIndex = endIndex[np.append(np.diff(endIndex)!=1,True)]
        for iSW in range(5):
            xPts = vecPixel[startIndex[iSW]:endIndex[iSW]+1]
            yPts = vecValue[startIndex[iSW]:endIndex[iSW]+1]
            interpFtn = InterpolatedUnivariateSpline(xPts,yPts,k=3)
            xInterp = np.array(range(swPts[iSW]-epLen,swPts[iSW+1]+epLen))
            yInterp = interpFtn(xInterp)
            ptsPixel['EW'+str(iSW+1)].append(xInterp)
            ptsLine['EW'+str(iSW+1)].append(np.ones_like(xInterp)*iLine)
            ptsValue['EW'+str(iSW+1)].append(yInterp)
    for iSW in range(5):
        pixels = np.stack(ptsPixel['EW'+str(iSW+1)])
        lines = np.stack(ptsLine['EW'+str(iSW+1)])
        values = np.stack(ptsValue['EW'+str(iSW+1)])
        gridSampleCoords = \
            np.array( (lines.flatten(),pixels.flatten()) ).transpose()
        gridExportPixels, gridExportLines = \
            np.meshgrid( range(np.min(pixels),np.max(pixels)+1),\
                         range(np.min(lines),np.max(lines)+1) )
        oLUT['pixel'].append(pixels[0])
        oLUT['EW'+str(iSW+1)] = \
            griddata( gridSampleCoords,values.flatten(),\
                      (gridExportLines,gridExportPixels), method='linear' )
    return oLUT


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
    # set fonts for Legend
    aaepFileName = os.path.join(os.path.dirname(
                                         os.path.realpath(__file__)),
                                         'AAEP_V20150722.npz')
    azimuthAntennaElementPattern = np.load(aaepFileName)

    def __getitem__(self, bandID):

        band = self.get_GDALRasterBand(bandID)
        name = band.GetMetadata().get('name', '')
        dataArray = Nansat.__getitem__(self, bandID)
        if name not in ['sigma0_HH', 'sigma0_HV', 'sigma0HH_', 'sigma0HV_']:
            return dataArray
        if name[-1]=='_':
            iPol = name[-3:-1]
        else:
            iPol = name[-2:]
        iSAFEimg = self.fileName

        addUpPower = -23.5
        filterOutPower = -25.0

        speedOfLight = 299792458.
        radarFrequency = 5405000454.33435
        azimuthSteeringRate = { 'EW1': 2.390895448 , 'EW2': 2.811502724, \
                                'EW3': 2.366195855 , 'EW4': 2.512694636, \
                                'EW5': 2.122855427                         }

        if iPol=='HH':
            ANNO_HEADER_XML = glob.glob(iSAFEimg+'/annotation/*001.xml')[0]
        elif iPol=='HV':
            ANNO_HEADER_XML = glob.glob(iSAFEimg+'/annotation/*002.xml')[0]
        xmldocElem = parse(ANNO_HEADER_XML)

        startTime = convertTime2Sec(getValue(xmldocElem,['startTime']))
        azimuthTimeInterval = float(
                            getValue(xmldocElem, ['azimuthTimeInterval']))
        numberOfSamples = int(getValue(xmldocElem, ['numberOfSamples']))
        numberOfLines = int(getValue(xmldocElem, ['numberOfLines']))

        orbit = { 'time':[], 'px':[], 'py':[], 'pz':[],\
                  'vx':[], 'vy':[], 'vz':[] }
        orbitList = getElem(xmldocElem,['orbitList'])
        for iOrbit in orbitList.getElementsByTagName('orbit'):
            orbit['time'].append(
                convertTime2Sec(getValue(iOrbit, ['time'])))
            orbit['px'].append(float(getValue(iOrbit, ['position','x'])))
            orbit['py'].append(float(getValue(iOrbit, ['position','y'])))
            orbit['pz'].append(float(getValue(iOrbit, ['position','z'])))
            orbit['vx'].append(float(getValue(iOrbit, ['velocity','x'])))
            orbit['vy'].append(float(getValue(iOrbit, ['velocity','y'])))
            orbit['vz'].append(float(getValue(iOrbit, ['velocity','z'])))

        azimuthFmRate = { 'azimuthTime':[], 't0':[], 'c0':[], 'c1':[], 'c2':[] }
        azimuthFmRateList = getElem(xmldocElem, ['azimuthFmRateList'])
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
        antPatList = getElem(xmldocElem,['antennaPattern','antennaPatternList'])
        for iAntPat in antPatList.getElementsByTagName('antennaPattern'):
            subswathID = getValue(iAntPat, ['swath'])
            antennaPatternTime[subswathID].append(
                convertTime2Sec(getValue(iAntPat, ['azimuthTime'])))

        geolocationGridPoint = { 'azimuthTime':[], 'slantRangeTime':[], \
                                 'line':[], 'pixel':[], 'elevationAngle':[] }
        geoGridPtList = getElem(xmldocElem,['geolocationGridPointList'])
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
        centerLineIndex = numberOfLines / 2
        wavelength = speedOfLight / radarFrequency

        replicaTime = convertTime2Sec(getValue(xmldocElem, ['replicaList',
                                                            'replica',
                                                            'azimuthTime']))
        zdtBias = (replicaTime - antennaPatternTime['EW1'][0]
                  + np.mean(np.diff(antennaPatternTime['EW1'])) / 2)

        noiseLut = getAnnoLut(self, iPol, 'noise')
        noiseLutInterp = restoreNoiseLUT(noiseLut)

        GRD = {}
        gridSampleCoord = np.array( (geolocationGridPoint['line'],
                                     geolocationGridPoint['pixel']) ).transpose()
        GRD['pixel'], GRD['line'] = np.meshgrid(range(numberOfSamples),
                                                range(numberOfLines))
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
        GRD['noiseLut'] = np.ones((numberOfLines, numberOfSamples)) * np.nan
        GRD['descallopingGain'] = np.ones((numberOfLines,
                                           numberOfSamples)) * np.nan

        swathMergeList = getElem(xmldocElem,['swathMergeList'])
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
                     for loopIdx in range(numberOfLines)])
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
                    indexShift = noiseLutInterp['pixel'][subswathIndex][0]
                    GRD['noiseLut'][iAziLine,
                                    firstRangeSample:lastRangeSample+1] = (
                            noiseLutInterp[subswathID][iAziLine,
                                                        firstRangeSample-indexShift:
                                                        lastRangeSample-indexShift+1])
                    GRD['descallopingGain'][iAziLine,
                                            firstRangeSample:lastRangeSample+1] = (
                          np.ones(lastRangeSample-firstRangeSample+1)
                        * 10**(ds[iAziLine]/10.))


        sigmaNought = getAnnoLut(self, iPol, 'calibration')
        sampleCoord = np.array( ( np.dot( np.array(sigmaNought['line'])[np.newaxis].T,
                                                   np.ones((1, len(sigmaNought['pixel'][0])))).flatten(),
                                                   np.stack(sigmaNought['pixel']).flatten() ) ).transpose()
        GRD['sigmaNought'] = np.ones((numberOfLines, numberOfSamples)) * np.nan
        GRD['sigmaNought'] = griddata(sampleCoord,
                                      np.stack(sigmaNought['value']).flatten(),
                                      (GRD['line'], GRD['pixel']),
                                      method='linear')

        DN = self['DN_'+iPol]
        DN[DN == 0] = np.nan
        if iPol=='HH':
            angularDependency = (
                  (0.60717 * np.exp(-0.12296 * GRD['elevationAngle']) + 0.02218) \
                - (0.60717 * np.exp(-0.12296 * 17.0) + 0.02218))
            sigma0 = ((DN**2-GRD['noiseLut']/GRD['descallopingGain'])
                     / GRD['sigmaNought']**2 - angularDependency)
        elif iPol=='HV':
            sigma0 = ((DN**2-GRD['noiseLut']/GRD['descallopingGain'])
                     / GRD['sigmaNought']**2 + 10**(addUpPower/10.))

        sigma0[sigma0 < 10**(filterOutPower/10.)] = np.nan
        return 10*np.log10(sigma0)
