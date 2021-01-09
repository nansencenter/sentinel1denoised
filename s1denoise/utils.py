import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import minimum_filter
from scipy.optimize import fminbound

def cost(x, pix_valid, interp, y_ref):
    """ Cost function for finding noise LUT shift in Range """
    y = interp(pix_valid+x)
    return 1 - pearsonr(y_ref, y)[0]

def fit_noise_scaling_coeff(meanS0, meanN0, pixelIndex):
    """ Fit noise scaling coefficient on one vector of sigma0, nesz and pixel values """
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
    return scalingFactor, correlationCoefficient, fitResidual

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

def get_DOM_subElement(element, tags):
    ''' Get sub-element from XML DOM element based on tags '''
    for tag in tags:
        element = element.getElementsByTagName(tag)[0]
    return element
