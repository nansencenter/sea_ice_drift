# Name:    seaicedrift.py
# Purpose: Container of SeaIceDrift class
# Authors:      Anton Korosov, Stefan Muckenhuber
# Created:      21.09.2016
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
from __future__ import absolute_import

import numpy as np

from nansat import Nansat

from sea_ice_drift.lib import get_n_img, get_drift_vectors
from sea_ice_drift.ftlib import feature_tracking
from sea_ice_drift.pmlib import pattern_matching

class SeaIceDrift(object):
    ''' Retrieve Sea Ice Drift using Feature Tracking and Pattern Matching'''
    def __init__(self, filename1, filename2, **kwargs):
        ''' Init from two file names '''
        self.filename1 = filename1
        self.filename2 = filename2

        # get Nansat and Image
        self.n1, self.img1 = get_n_img(self.filename1, **kwargs)
        self.n2, self.img2 = get_n_img(self.filename2, **kwargs)


    def get_drift_FT(self, **kwargs):
        ''' Get sea ice drift using Feature Tracking '''
        x1, y1, x2, y2 = feature_tracking(self.n1, self.img1,
                                          self.n2, self.img2, **kwargs)
        return get_drift_vectors(self.n1, x1, y1,
                                 self.n2, x2, y2, **kwargs)
    

    def get_drift_PM(self, lons, lats, lon1, lat1, lon2, lat2, **kwargs):
        ''' Get sea ice drift using Pattern Matching '''
        x1, y1 = self.n1.transform_points(lon1, lat1, 1)
        x2, y2 = self.n2.transform_points(lon2, lat2, 1)
        return pattern_matching(lons, lats,
                                self.n1, self.img1, x1, y1,
                                self.n2, self.img2, x2, y2)
