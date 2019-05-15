from setuptools import setup, find_packages

setup(
    name='s1denoise',
    version='0.1',
    description='Thermal noise correction of Sentinel-1 TOPS GRDM EW products ',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Utilities'
    ],
    keywords='thermal noise correction sentinel',
    author='Jeong-Won Park, Anton Korosov',
    author_email='jeong-won.park@nersc.no, anton.korosov@nersc.no',
    packages=find_packages(),
    scripts=['s1denoise/scripts/s1_thermal_denoise.py'],
    package_data={'s1denoise' :['*.npz']},
    install_requires=[
        'nansat',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False)
