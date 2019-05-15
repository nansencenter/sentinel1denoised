FROM akorosov/nansat
LABEL maintainer="Anton Korosov <anton.korosov@nersc.no>"
LABEL purpose="Python lib for removal thermal noise from Sentinel-1 TOPSAR"

ENV CPL_ZIP_ENCODING=UTF-8

COPY s1denoise /tmp/s1denoise
COPY setup.py /tmp/
WORKDIR /tmp
RUN python setup.py install

