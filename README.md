[![Build Status](https://travis-ci.org/nansencenter/sea_ice_drift.svg?branch=master)](https://travis-ci.org/nansencenter/sea_ice_drift)
[![Coverage Status](https://coveralls.io/repos/nansencenter/sea_ice_drift/badge.svg?branch=master)](https://coveralls.io/r/nansencenter/sea_ice_drift)

# sea_ice_drift
Sea ice drift from satellite data using OpenCV feature tracking

A computational efficient, open source feature tracking algorithm, called ORB, is adopted and tuned for sea ice drift retrieval from Sentinel-1 SAR images.

## Requirements:
 * Nansat (https://github.com/nansencenter/nansat, scientist friendly open-source Python toolbox for processing 2D satellite earth observation data)
 * openCV (http://opencv.org, open-source computer vision)

## Installation
 * Download the code and run `python setup.py install`

## Usage
```
# import relvant libraries
import matplotlib.pyplot as plt
from nansat import Nansat
from sea_ice_drift import SeaIceDrift

# open pair of satellite images using Nansat and SeaIceDrift
n1 = SeaIceDrift(filename1)
n2 = Nansat(filename1)

# run detection and matching of features on two images
u, v, lon1, lat1, lon2, lat2 = n1.get_drift_vectors(n2, nFeatures=10000, maxSpeed=0.5)

# plot
plt.quiver(lon1, lat1, u, v)

```

## Example
 * Download the following Sentinel-1 files from https://scihub.esa.int:
   * S1A_EW_GRDM_1SDH_20150328T074433_20150328T074533_005229_0069A8_801E.zip
   * S1A_EW_GRDM_1SDH_20150329T163452_20150329T163552_005249_006A15_FD89.zip
 * Run `example.py`
