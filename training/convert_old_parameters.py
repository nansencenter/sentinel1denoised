""" Convert old JSON files to a new format (needed only once) """

import json

param_types = {
    'powerBalancingParameters': 'PB',
    'noiseScalingParameters': 'NS',
    'noiseVarianceParameters': 'NV',
    'extraScalingParameters': 'ES',
}

swath2res = {
    'IW': 'GRDH',
    'EW': 'GRDM',
}

new_par = {}
for platform in ['S1A', 'S1B']:
    ifile = f'denoising_parameters_{platform}.json'

    with open(ifile, 'rt') as f:
        old_par = json.load(f)

    for pol in old_par:
        for tp in old_par[pol]:
            param_type = param_types[tp]
            if param_type in ['NS', 'PB']:
                for swath in old_par[pol][tp]:
                    for ipf in old_par[pol][tp][swath]:
                        mode = swath[:2]
                        res = swath2res[mode]
                        new_key = f'{platform}_{mode}_{res}_{pol}_{param_type}_{ipf}'
                        if new_key not in new_par:
                            new_par[new_key] = {}
                        new_par[new_key][swath] = old_par[pol][tp][swath][ipf]
            elif param_type in ['NV', 'ES']:
                ipf = '2.9'
                mode = 'EW'
                res = swath2res[mode]
                for swath in old_par[pol][tp]:
                    new_key = f'{platform}_{mode}_{res}_{pol}_{param_type}_{ipf}'
                    if new_key not in new_par:
                        new_par[new_key] = {}
                    new_par[new_key][swath] = old_par[pol][tp][swath]

with open('../denoising_parameters_old.json', 'w') as f:
    json.dump(new_par, f)
