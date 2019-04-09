FROM akorosov/nansat
LABEL maintainer="Anton Korosov <anton.korosov@nersc.no>"
LABEL purpose="Python lib for removal thermal noise from Sentinel-1 TOPSAR"
COPY s1denoise /opt/conda/lib/python3.7/site-packages/s1denoise
