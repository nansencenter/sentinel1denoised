import numpy as np
import imp


def get_denoising_coeffs(preScalingFactor,polarization,IPFversion):
    
    if (sum(preScalingFactor) != 5) and (IPFversion < 2.53):
        if polarization == 'HH':
            noiseScalingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
            powerBalancingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
        elif polarization == 'HV':
            noiseScalingCoeff = [
                1.20928121,  0.95343420,  1.03134686,  0.98194383,  0.91263135 ]
            powerBalancingCoeff = [
                +1.9399e-04,  +1.5528e-04,  +7.2287e-05,  +4.3939e-05,  +3.4226e-05 ]

    elif (sum(preScalingFactor) == 5) and (2.50 <= IPFversion < 2.60):
        if polarization == 'HH':
            noiseScalingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
            powerBalancingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
        elif polarization == 'HV':
            noiseScalingCoeff = [
                1.19712728,  0.94171422,  1.01746497,  0.97366257,  0.90949900 ]
            powerBalancingCoeff = [
                +2.7159e-4,  +2.8389e-4,  +2.1187e-4,  +1.9654e-4,  +1.9376e-4 ]

    elif (sum(preScalingFactor) == 5) and (2.60 <= IPFversion < 2.70):
        
        if polarization == 'HH':
            noiseScalingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
            powerBalancingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
        elif polarization == 'HV':
            noiseScalingCoeff = [
                1.13334859,  0.93294159,  0.96018289,  0.91170991,  0.84259001 ]
            powerBalancingCoeff = [
                +2.0795e-4,  +2.4768e-4,  +1.8771e-4,  +1.6059e-4,  +1.5271e-4 ]

    elif (sum(preScalingFactor) == 5) and (2.70 <= IPFversion < 2.80):
        
        if polarization == 'HH':
            noiseScalingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
            powerBalancingCoeff = [
                0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
        elif polarization == 'HV':
            noiseScalingCoeff = [
                1.36265638,  0.99054314,  1.04277516,  0.98972426,  0.93180422 ]
            powerBalancingCoeff = [
                +3.1022e-4,  +3.5531e-4,  +3.1613e-4,  +3.0889e-4,  +3.0555e-4 ]

    else:
        noiseScalingCoeff = [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
        powerBalancingCoeff = [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
        print( "\nWARNING: noise scaling coefficients are not defined "
               "for IPF version %s." % IPFversion )

    extraScalingCoeff = np.load( imp.find_module('sentinel1denoised')[1]
                                 + '/extraNoiseScalingData.npz' )

    return noiseScalingCoeff,powerBalancingCoeff,extraScalingCoeff
