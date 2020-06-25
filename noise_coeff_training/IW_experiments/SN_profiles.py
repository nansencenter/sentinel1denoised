import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from s1denoise.S1_TOPS_GRD_NoiseCorrection import Sentinel1Image
from multiprocessing import Pool
import sys
import glob
import os

def proc(s1_filename):
    n = Sentinel1Image('%s'
                       % (s1_filename))
    results = {}
    results['src'] = os.path.basename(s1_filename)
    results['inc'] = np.nanmean(n.incidenceAngleMap(polarization='HV'), axis=0)
    sz = n.rawSigma0Map(polarization='HV')
    sz[sz==0] = np.nan
    results['sz'] = np.nanmean(sz, axis=0)
    results['nesz_esa'] = np.nanmean(n.rawNoiseEquivalentSigma0Map(polarization='HV'), axis=0)
    results['nesz_nersc'] = np.nanmean(n.modifiedNoiseEquivalentSigma0Map(polarization='HV'), axis=0)
    return results

# Save npz files?
f_save_npz = False

# Titles and names
orbit_desc = 'Descending relative orbit #127 (IPF 3.x)'
orbit_type = 'dsc_127'
fname_pref = '%s_s0_denoised' % orbit_type
instrument = 'S1A'

f_path = sys.argv[1]
s1_files = glob.glob('%s/*%s*.zip' % (f_path, instrument))#[0:10]

print('\n%s files found\n' % len(s1_files))

p = Pool(4)
results = p.map(proc, s1_files)
p.close(); p.terminate(); p.join();

###################
# Denoised Sigma0
###################

plt.clf()
plt.figure(figsize=(16,12))
plt.ylim(ymax=0.006)
for r in results:
    data = r['sz']-r['nesz_esa']
    # Only s0 < 0.005
    if np.nanmax(data) < 0.004:
        #if not len(data[data > 0.001]) > 0:
        plt.plot(r['inc'], data, label=r['src'], linewidth=0.5)

plt.title('%s. Sigma0-NESZ(ESA)' % orbit_desc, fontsize=24)
plt.tight_layout()
plt.legend()
ax = plt.axis()
plt.savefig('%s_ln_ESA_profiles.png' % fname_pref, bbox_inches='tight', dpi=300)

plt.clf()
plt.ylim(ymax=0.006)
for r in results:
    data=r['sz']-r['nesz_nersc']
    if np.nanmax(data) < 0.004:
    #if not len(data[data > 0.001])>0:
        plt.plot(r['inc'], data, label=r['src'], linewidth=0.5)
plt.title('%s. Sigma0-NESZ(NERSC)' % orbit_desc, fontsize=24)
plt.tight_layout()
plt.legend()
plt.axis(ax)
plt.savefig('%s_ln_NERSC_profiles.png' % fname_pref, bbox_inches='tight', dpi=300)

########
# NESZ
########
"""
fname_pref = '%s_nesz_esa' % orbit_type
plt.clf()
plt.figure(figsize=(16,12))
plt.ylim(ymax=0.008)
for r in results:
    data = r['nesz_esa']
    data_sz = r['sz']-r['nesz_nersc']
    #if not len(data_sz[data_sz > 0.001]) > 0:
    #data[data>np.nanmean(data)*2]=np.nan
    plt.plot(r['inc'], data, label=r['src'], linewidth=0.5)
plt.title('%s. NESZ(ESA)' % orbit_desc, fontsize=24)
plt.tight_layout()
plt.legend()
ax = plt.axis()
plt.savefig('%s_ln_NESZ_ESA_profiles.png' % fname_pref, bbox_inches='tight', dpi=300)

fname_pref = '%s_nesz_nersc' % orbit_type
plt.clf()
plt.figure(figsize=(16,12))
plt.ylim(ymax=0.008)
for r in results:
    data = r['nesz_nersc']
    #data[data>np.nanmean(data)*2]=np.nan
    data_sz = r['sz']-r['nesz_nersc']
    #if not len(data_sz[data_sz > 0.001]) > 0:
    plt.plot(r['inc'], data, label=r['src'], linewidth=0.5)
plt.title('%s. NESZ(NERSC)' % orbit_desc, fontsize=24)
plt.tight_layout()
plt.legend()
ax = plt.axis()
plt.savefig('%s_ln_NESZ_NERSC_profiles.png' % fname_pref, bbox_inches='tight', dpi=300)
"""
#############
# Save NPZ
#############
#outfile = TemporaryFile()

if f_save_npz:
    npz_path = 'npz'
    try:
        os.makedirs(npz_path)
    except:
        pass

    for r in results:
        np.savez('%s/inc_%s' % (npz_path, r['src']), inc=r['inc'])
        #_ = outfile.seek(0)
        np.savez('%s/s0_%s' % (npz_path, r['src']), s0=r['sz'])
        np.savez('%s/nesz_ESA_%s' % (npz_path, r['src']), nesz_ESA=r['nesz_esa'])
        np.savez('%s/nesz_NERSC_%s' % (npz_path, r['src']), nesz_NERSC=r['nesz_nersc'])

