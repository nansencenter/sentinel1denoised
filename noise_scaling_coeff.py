def noise_scaling(noisePowerPreScalingFactor,polarization,IPFversion):
    
    if (noisePowerPreScalingFactor != 1.0) and (IPFversion < 2.53):
        noiseScalingCoeff = [ 1.42,  1.14,  0.98,  0.95,  0.83 ]
        if polarization == 'HH':
            balancingPower = [ 0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00 ]
        elif polarization == 'HV':
            balancingPower = [ -1.46e-03,  +1.60e-06,  0.00e+00,  +1.88e-04,  +3.49e-04 ]
    elif (noisePowerPreScalingFactor == 1.0) and (2.50 <= IPFversion < 2.60):
        noiseScalingCoeff = [ 1.20,  0.92,  1.00,  0.95,  0.88 ]
        if polarization == 'HH':
            balancingPower = [ +9.51e-03,  +5.66e-03,  0.00e+00,  -5.31e-03,  -4.62e-03 ]
        elif polarization == 'HV':
            balancingPower = [ -1.50e-03,  -8.04e-06,  0.00e+00,  +1.83e-04,  +3.29e-04 ]
    elif (noisePowerPreScalingFactor == 1.0) and (2.60 <= IPFversion < 2.70):
        noiseScalingCoeff = [ 1.15,  0.92,  0.95,  0.89,  0.81 ]
        if polarization == 'HH':
            balancingPower = [ +5.87e-03,  +2.55e-03,  0.00e+00,  -4.50e-03,  -5.72e-03 ]
        elif polarization == 'HV':
            balancingPower = [ -1.37e-03,  -1.02e-04,  0.00e+00,  +1.39e-04,  +2.69e-04 ]
    elif (noisePowerPreScalingFactor == 1.0) and (2.70 <= IPFversion < 2.80):
        noiseScalingCoeff = [ 1.41,  0.97,  1.04,  0.97,  0.88 ]
        if polarization == 'HH':
            balancingPower = [ +6.75e-04,  +2.05e-03,  0.00e+00,  -3.45e-03,  -3.95e-03 ]
        elif polarization == 'HV':
            balancingPower = [ -1.82e-03,  -1.05e-04,  0.00e+00,  +1.52e-04,  +2.70e-04 ]
    else:
        noiseScalingCoeff = [ 1.0 , 1.0 , 1.0 , 1.0 , 1.0 ]
        balancingPower = [ 0 , 0 , 0 , 0 , 0 ]
        print( "\nWARNING: noise scaling coefficients are not defined for IPF version %s."
              "\n         please contact jeong-won.park@nersc.no for requesting update.\n" % IPFversion )
    return noiseScalingCoeff,balancingPower