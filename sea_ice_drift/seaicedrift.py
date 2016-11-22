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

from sea_ice_drift.lib import (get_n_img,
                               get_denoised_object,
                               reproject_gcp_to_stere,
                               get_uint8_image,
                               get_displacement_km)
from sea_ice_drift.ftlib import (find_key_points,
                                 domain_filter,
                                 get_match_coords,
                                 max_drift_filter,
                                 lstsq_filter)

class SeaIceDrift(object):
    ''' Retrieve Sea Ice Drift using Feature Tracking and Pattern Matching'''
    def __init__(self, filename1, filename2):
        ''' Init from two file names '''
        self.filename1 = filename1
        self.filename2 = filename2

    def feature_tracking(self, bandName='sigma0_HV',
                          factor=0.5, vmin=0, vmax=0.013,
                          domainMargin=10, maxDrift=20,
                          denoise=False, dB=False, **kwargs):
        # get Nansat and Image
        n1, img1 = get_n_img(self.filename1, bandName, factor,
                             vmin, vmax, denoise, dB, **kwargs)
        n2, img2 = get_n_img(self.filename2, bandName, factor,
                             vmin, vmax, denoise, dB, **kwargs)

        # find many key points
        kp1, descr1 = find_key_points(img1, **kwargs)
        kp2, descr2 = find_key_points(img2, **kwargs)

        # filter keypoints by Domain
        kp1, descr1 = domain_filter(n1, kp1, descr1, n2, domainMargin)
        kp2, descr2 = domain_filter(n2, kp2, descr2, n1, domainMargin)

        # find coordinates of matching key points
        x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2, **kwargs)

        # filter out pair with too high drift
        x1, y1, x2, y2 = max_drift_filter(n1, x1, y1, n2, x2, y2, maxDrift)

        # filter out inconsistent pairs
        x1, y1, x2, y2 = lstsq_filter(x1, y1, x2, y2, **kwargs)

        return n1, img1, x1, y1, n2, img2, x2, y2

    def get_drift_vectors(self, n1, x1, y1, n2, x2, y2, ll2km='domain'):
        # convert x,y to lon, lat
        lon1, lat1 = n1.transform_points(x1, y1)
        lon2, lat2 = n2.transform_points(x2, y2)

        # find displacement in kilometers
        u, v = get_displacement_km(n1, x1, y1, n2, x2, y2, ll2km=ll2km)

        # convert to speed in m/s
        dt = n2.time_coverage_start - n1.time_coverage_start
        u = u * 1000 / dt.total_seconds()
        v = v * 1000 / dt.total_seconds()

        return u, v, lon1, lat1, lon2, lat2
