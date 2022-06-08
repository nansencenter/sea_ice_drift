[![Build Status](https://travis-ci.org/nansencenter/sea_ice_drift.svg?branch=master)](https://travis-ci.org/nansencenter/sea_ice_drift)
[![Coverage Status](https://coveralls.io/repos/nansencenter/sea_ice_drift/badge.svg?branch=master)](https://coveralls.io/r/nansencenter/sea_ice_drift)
[![DOI](https://zenodo.org/badge/46479183.svg)](https://zenodo.org/badge/latestdoi/46479183)

## Sea ice drift from Sentinel-1 SAR data

A computationally efficient, open source feature tracking algorithm,
called ORB, is adopted and tuned for retrieval of the first guess
sea ice drift from Sentinel-1 SAR images. Pattern matching algorithm
based on MCC calculation is used further to retrieve sea ice drift on a
regular grid.

## References:
 * Korosov A.A. and Rampal P., A Combination of Feature Tracking and Pattern Matching with Optimal Parametrization for Sea Ice Drift Retrieval from SAR Data, Remote Sens. 2017, 9(3), 258; [doi:10.3390/rs9030258](http://www.mdpi.com/2072-4292/9/3/258)
 * Muckenhuber S., Korosov A.A., and Sandven S., Open-source feature-tracking algorithm for sea ice drift retrieval from Sentinel-1 SAR imagery, The Cryosphere, 10, 913-925, [doi:10.5194/tc-10-913-2016](http://www.the-cryosphere.net/10/913/2016/), 2016

## Running with Docker
```
# run ipython with SeaIceDrift
docker run --rm -it -v /path/to/data:/home/jovyan/work nansencenter/seaicedrift ipython

# run jupyter notebook with SeaIceDrift
docker run --rm -p 8888:8888 -v /path/to/data/and/notebooks:/home/jovyan/work nansencenter/seaicedrift
```

## Installation on Ubuntu
```
# install some requirements with apt-get
apt-get install -y --no-install-recommends libgl1-mesa-glx gcc build-essential

# install some requirements with conda
conda install -c conda-forge gdal cartopy opencv

# install other requirements with pip
pip install netcdf4 nansat

# clone code
git clone https://github.com/nansencenter/sea_ice_drift.git
cd sea_ice_drift

# install SeaIceDrift
python setup.py install
```

## Usage example
```
# download example datasets
wget https://github.com/nansencenter/sea_ice_drift_test_files/raw/master/S1B_EW_GRDM_1SDH_20200123T120618.tif
wget https://github.com/nansencenter/sea_ice_drift_test_files/raw/master/S1B_EW_GRDM_1SDH_20200125T114955.tif

# start Python and import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from nansat import Nansat
from sea_ice_drift import SeaIceDrift

# open pair of satellite images using Nansat and SeaIceDrift
filename1='S1B_EW_GRDM_1SDH_20200123T120618.tif'
filename2='S1B_EW_GRDM_1SDH_20200125T114955.tif'
sid = SeaIceDrift(filename1, filename2)

# run ice drift retrieval using Feature Tracking
uft, vft, lon1ft, lat1ft, lon2ft, lat2ft = sid.get_drift_FT()

# plot
plt.quiver(lon1ft, lat1ft, uft, vft);plt.show()

# define a grid (e.g. regular)
lon1pm, lat1pm = np.meshgrid(np.linspace(-33.5, -30.5, 50),
                             np.linspace(83.6, 83.9, 50))

# run ice drift retrieval for regular points using Pattern Matching
# use results from the Feature Tracking as the first guess
upm, vpm, apm, rpm, hpm, lon2pm, lat2pm = sid.get_drift_PM(
        lon1pm, lat1pm,
        lon1ft, lat1ft,
        lon2ft, lat2ft)
# select high quality data only
gpi = rpm*hpm > 4

# plot high quality data on a regular grid
plt.quiver(lon1pm[gpi], lat1pm[gpi], upm[gpi], vpm[gpi], rpm[gpi])

```
Full example [here](https://github.com/nansencenter/sea_ice_drift/blob/master/examples/simple.py)

![Feature Tracking and the first SAR image](https://raw.githubusercontent.com/nansencenter/sea_ice_drift/master/examples/sea_ice_drift_FT_img1.png)

![Pattern Matching and the second SAR image](https://raw.githubusercontent.com/nansencenter/sea_ice_drift/master/examples/sea_ice_drift_PM_img2.png)
