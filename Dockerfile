FROM akorosov/nansat-lectures

RUN apt-get update \
&&  apt-get install -y --no-install-recommends libgl1-mesa-glx \
&&  conda install -y opencv \
&&  conda clean -a -y \
&&  rm /opt/conda/pkgs/* -rf

COPY *.py /tmp/
COPY sea_ice_drift /tmp/sea_ice_drift
WORKDIR /tmp
RUN python setup.py install
WORKDIR /src
