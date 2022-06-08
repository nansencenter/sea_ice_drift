FROM jupyter/minimal-notebook

USER root

RUN apt-get update \
&&  apt-get install -y --no-install-recommends libgl1-mesa-glx gcc build-essential \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN conda install gdal cartopy opencv \
&&  conda clean -afy \
&&  pip install --no-cache-dir \
    netcdf4 \
    Pillow \
    python-dateutil \
    PyYAML \
    requests \
    urllib3 \
    xdg    

ENV PROJ_LIB=/opt/conda/share/proj

COPY py-thesaurus-interface /tmp/py-thesaurus-interface
WORKDIR /tmp/py-thesaurus-interface
RUN python setup.py install

RUN pip install nansat --target /tmp/install_nansat

COPY *.py /tmp/
COPY sea_ice_drift /tmp/sea_ice_drift
WORKDIR /tmp
RUN python setup.py install

USER jovyan
WORKDIR /home/jovyan
RUN python -c 'import pythesint as pti; pti.update_all_vocabularies()'

ENV MOD44WPATH=/home/jovyan/MOD44W
RUN wget -nc -nv -P $MOD44WPATH https://github.com/nansencenter/mod44w/raw/master/MOD44W.tgz \
&&  tar -xzf $MOD44WPATH/MOD44W.tgz -C $MOD44WPATH/ \
&&  rm $MOD44WPATH/MOD44W.tgz
