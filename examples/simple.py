# Name:    simple.py
# Purpose: Simple example of SeaIceDrift application
# Authors:      Anton Korosov
# Created:      26.08.2016
# Copyright:    (c) NERSC 2016
# Licence:
# This file is part of SeaIceDrift.
# SeaIceDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# http://www.gnu.org/licenses/gpl-3.0.html
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
import os
import sys
import glob
import unittest
import inspect

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from nansat import Nansat, Domain, NSR

from sea_ice_drift import SeaIceDrift

# download Sentinel-1 data from https://scihub.copernicus.eu/dhus
# or get small size sample files here:
# wget https://github.com/nansencenter/sea_ice_drift_test_files/raw/master/S1B_EW_GRDM_1SDH_20200123T120618.tif
# wget https://github.com/nansencenter/sea_ice_drift_test_files/raw/master/S1B_EW_GRDM_1SDH_20200125T114955.tif
# ==== ICE DRIFT RETRIEVAL ====

# define border for region of interest
lone = -33.5
lonw = -30.5
lats = 83.6
latn = 83.9

# open files, read 'sigma0_HV' band and convert to UInt8 image
f1 = 'S1B_EW_GRDM_1SDH_20200123T120618.tif'
f2 = 'S1B_EW_GRDM_1SDH_20200125T114955.tif'
sid = SeaIceDrift(f1, f2)

# apply Feature Tracking algorithm and retrieve ice drift speed
# and starting/ending coordinates of matched keypoints
uft, vft, lon1ft, lat1ft, lon2ft, lat2ft = sid.get_drift_FT()

# user defined grid of points:
lon1pm, lat1pm = np.meshgrid(np.linspace(lone, lonw, 50),
                     np.linspace(lats, latn, 50))

# apply Pattern Matching and find sea ice drift speed
# for the given grid of points
upm, vpm, apm, rpm, hpm, lon2pm, lat2pm = sid.get_drift_PM(
                                    lon1pm, lat1pm,
                                    lon1ft, lat1ft,
                                    lon2ft, lat2ft)

# ==== PLOTTING ====
# get coordinates of SAR scene borders
lon1, lat1 = sid.n1.get_border()
lon2, lat2 = sid.n2.get_border()

# prepare projected images with sigma0_HV
d = Domain(NSR().wkt, '-te %f %f %f %f -ts 500 500' % (lone, lats, lonw, latn))
sid.n1.reproject(d)
s01 = sid.n1['sigma0_HV']
sid.n2.reproject(d)
s02 = sid.n2['sigma0_HV']

# plot the projected image from the first SAR scene
plt.imshow(s01, extent=[lone, lonw, lats, latn], cmap='gray', aspect=10)
# plot vectors of sea ice drift from Feature Tracking
plt.quiver(lon1ft, lat1ft, uft, vft, color='r',
           angles='xy', scale_units='xy', scale=0.2)
# plot border of the second SAR scene
plt.plot(lon2, lat2, '.-r')
# set X/Y limits of figure
plt.xlim([lone, lonw])
plt.ylim([lats, latn])
plt.savefig('sea_ice_drift_FT_img1.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close('all')

# plot the projected image from the second SAR scene
plt.imshow(s02, extent=[lone, lonw, lats, latn], cmap='gray', aspect=10)
# filter only high quality pixels
gpi = rpm*hpm > 4
# plot vectors of sea ice drift from Feature Tracking, color by MCC
plt.quiver(lon1pm[gpi], lat1pm[gpi], upm[gpi], vpm[gpi], rpm[gpi],
           angles='xy', scale_units='xy', scale=0.2)
# plot border of the first SAR scene
plt.plot(lon1, lat1, '.-r')
# set X/Y limits of figure
plt.xlim([lone, lonw])
plt.ylim([lats, latn])
plt.savefig('sea_ice_drift_PM_img2.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close('all')
