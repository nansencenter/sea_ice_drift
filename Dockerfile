FROM jupyter/minimal-notebook

USER root

RUN apt-get update \
&&  apt-get install -y --no-install-recommends libgl1-mesa-glx \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN conda install gdal cartopy opencv \
&&  conda clean -afy \
&&  pip install netcdf4 --no-cache-dir

COPY py-thesaurus-interface /tmp/py-thesaurus-interface
WORKDIR /tmp/py-thesaurus-interface
RUN python setup.py install

COPY nansat /tmp/nansat
WORKDIR /tmp/nansat
RUN python setup.py install

COPY *.py /tmp/
COPY sea_ice_drift /tmp/sea_ice_drift
WORKDIR /tmp
RUN python setup.py install

USER jovyan
WORKDIR /home/jovyan
RUN python -c 'import pythesint as pti; pti.update_all_vocabularies()'
