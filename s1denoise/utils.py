from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import fminbound
from scipy.ndimage.morphology import distance_transform_edt

def cubic_hermite_interpolation(x,y,xi):
    ''' Get interpolated value for given time '''
    return np.polynomial.hermite.hermval(xi, np.polynomial.hermite.hermfit(x,y,deg=3))

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

def fill_gaps(array, mask, distance=15):
    """ Fill gaps in input raster

    Parameters
    ----------
    array : 2D numpy.array
        Ratser with deformation field
    mask : 2D numpy.array
        Where are gaps
    distance : int
        Minimum size of gap to fill

    Returns
    -------
    arra : 2D numpy.array
        Ratser with gaps filled

    """
    dist, indi = distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True)
    gpi = dist <= distance
    r,c = indi[:,gpi]
    array[gpi] = array[r,c]
    return array

def skip_swath_borders(swath_ids, skip=1):
    swath_ids_skip = []
    for swid in swath_ids:
        swid_skip = np.array(swid)
        for j in range(1,6):
            gpi = np.where(swid == j)[0]
            if gpi.size > 0:
                swid_skip[gpi[:skip]] = 0
                swid_skip[gpi[-skip:]] = 0
        swath_ids_skip.append(swid_skip)
    return swath_ids_skip

def build_AY_matrix(swath_ids, sigma0hv, apg, incang, s0hv_max, s0hv_apg_corr_min):
    "HV = 1, IA, 1, APG1, 1, APG2, 1, APG3, 1, APG4, 1, APG5"

    A_123 = []                  # [1]
    A_apg = defaultdict(list)   # [APG scale and 1]
    Y = []                      # [HV]

    for iswath in range(1,6):
        for swath_ids_v, sigma0hv_v, apg_v, incang_v in zip(swath_ids, sigma0hv, apg, incang):
            gpi = np.where((swath_ids_v == iswath) * (np.isfinite(sigma0hv_v*apg_v)))[0]
            # append only small enough values of HV
            if (
                (np.nanmean(sigma0hv_v[gpi]) < s0hv_max[iswath]) and 
                (pearsonr(sigma0hv_v[gpi], apg_v[gpi])[0] > s0hv_apg_corr_min[iswath])
            ):
                A_123.append([
                    np.ones(gpi.size),
                    incang_v[gpi],
                ])
                A_apg[iswath].append(np.vstack([
                    np.ones(gpi.size),
                    apg_v[gpi],
                ]).T)
                Y.append(sigma0hv_v[gpi])

    if len(A_123) == 0:
        return None, None
    A_123 = np.hstack(A_123).T
    Y = np.hstack(Y)[None].T

    A = []
    for iswath in A_apg:
        A_apg_stack = np.vstack(A_apg[iswath])
        A_apg_all = np.zeros((A_apg_stack.shape[0], 10))
        A_apg_all[:, int((iswath-1)*2):int((iswath-1)*2+2)] = A_apg_stack
        A.append(A_apg_all)

    A = np.hstack([np.vstack(A), A_123])
    gpi = np.isfinite(A.sum(axis=1)) * np.isfinite(Y.flat)

    A = A[gpi]
    Y = Y[gpi]

    return A, Y

def solve(A, Y):
    B = np.linalg.lstsq(A, Y, rcond=None)
    Y_rec = np.dot(A, B[0])
    rmsd = np.sqrt(np.mean((Y - Y_rec)**2))
    return B[0].flatten(), rmsd
