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

from sea_ice_drift.lib import get_n, get_drift_vectors
from sea_ice_drift.ftlib import feature_tracking
from sea_ice_drift.pmlib import pattern_matching

class SeaIceDrift(object):
    ''' Retrieve Sea Ice Drift using Feature Tracking and Pattern Matching'''
    def __init__(self, filename1, filename2, **kwargs):
        ''' Initialize from two file names:
        Open files with Nansat
        Read data from sigma0_HV or other band and convert to UInt8
        Parameters
        ----------
            filename1 : str, file name of the first Sentinel-1 image
            filename2 : str, file name of the second Sentinel-1 image
        '''
        self.filename1 = filename1
        self.filename2 = filename2

        # get Nansat
        self.n1 = get_n(self.filename1, **kwargs)
        self.n2 = get_n(self.filename2, **kwargs)

    def get_drift_FT(self, **kwargs):
        ''' Get sea ice drift using Feature Tracking
        Parameters
        ----------
            **kwargs : parameters for
                feature_tracking
                get_drift_vectors
        Returns
        -------
            u : 1D vector - eastward ice drift speed
            v : 1D vector - northward ice drift speed
            lon1 : 1D vector - longitudes of source points
            lat1 : 1D vector - latitudes of source points
            lon2 : 1D vector - longitudes of destination points
            lat2 : 1D vector - latitudes of destination points
        '''
        x1, y1, x2, y2 = feature_tracking(self.n1, self.n2, **kwargs)
        return get_drift_vectors(self.n1, x1, y1,
                                 self.n2, x2, y2, **kwargs)


    def get_drift_PM(self, lons, lats, lon1, lat1, lon2, lat2, **kwargs):
        ''' Get sea ice drift using Pattern Matching
        Parameters
        ----------
            lons : 1D vector, longitude of result vectors on image 1
            lats : 1D vector, latitude of result  vectors on image 1
            lon1 : 1D vector, longitude of keypoints on image1
            lat1 : 1D vector, latitude  of keypoints on image1
            lon2 : 1D vector, longitude of keypoints on image2
            lat2 : 1D vector, latitude  of keypoints on image2
            **kwargs : parameters for
                feature_tracking
                get_drift_vectors
        Returns
        -------
            u : 1D vector, eastward ice drift speed, m/s
            v : 1D vector, eastward ice drift speed, m/s
            a : 1D vector, angle that gives the highes MCC
            r : 1D vector, MCC
            h : 1D vector, Hessian of CC matrix and MCC point
            lon2_dst : 1D vector, longitude of results on image 2
            lat2_dst : 1D vector, latitude  of results on image 2
        '''
        x1, y1 = self.n1.transform_points(lon1, lat1, 1)
        x2, y2 = self.n2.transform_points(lon2, lat2, 1)
        return pattern_matching(lons, lats, self.n1, x1, y1,
                                            self.n2, x2, y2, **kwargs)
