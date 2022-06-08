# Name:    ftlib.py
# Purpose: Container of Feature Tracking functions
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
from __future__ import absolute_import, print_function

import time
import numpy as np

import cv2

from sea_ice_drift.lib import (get_speed_ms,
                               interpolation_poly,
                               get_displacement_km)

def find_key_points(image,
                    edgeThreshold=34,
                    nFeatures=100000,
                    nLevels=7,
                    patchSize=34,
                    verbose=False,
                    **kwargs):
    ''' Initiate detector and find key points on an image
    Parameters
    ----------
        image : 2D UInt8 Numpy array - image
        edgeThreshold : int - parameter for OpenCV detector
        nFeatures : int - parameter for OpenCV detector
        nLevels : int - parameter for OpenCV detector
        patchSize : int - parameter for OpenCV detector
    Returns
    -------
        keyPoints : list - coordinates of keypoint on image
        descriptors : list - binary descriptos of kepoints
    '''
    if cv2.__version__.startswith('3.') or cv2.__version__.startswith('4.'):
        detector = cv2.ORB_create()
        detector.setEdgeThreshold(edgeThreshold)
        detector.setMaxFeatures(nFeatures)
        detector.setNLevels(nLevels)
        detector.setPatchSize(patchSize)
    else:
        detector = cv2.ORB()
        detector.setInt('edgeThreshold', edgeThreshold)
        detector.setInt('nFeatures', nFeatures)
        detector.setInt('nLevels', nLevels)
        detector.setInt('patchSize', patchSize)
    keyPoints, descriptors = detector.detectAndCompute(image, None)
    if verbose:
        print('Key points found: %d' % len(keyPoints))
    return keyPoints, descriptors


def get_match_coords(keyPoints1, descriptors1,
                                    keyPoints2, descriptors2,
                                    matcher=cv2.BFMatcher,
                                    norm=cv2.NORM_HAMMING,
                                    ratio_test=0.7,
                                    verbose=False,
                                    **kwargs):
    ''' Filter matching keypoints and convert to X,Y coordinates
    Parameters
    ----------
        keyPoints1 : list - keypoints on img1 from find_key_points()
        descriptors1 : list - descriptors on img1 from find_key_points()
        keyPoints2 : list - keypoints on img2 from find_key_points()
        descriptors2 : list - descriptors on img2 from find_key_points()
        matcher : matcher from CV2
        norm : int - type of distance
        ratio_test : float - Lowe ratio
        verbose : bool - print some output ?
    Returns
    -------
        x1, y1, x2, y2 : coordinates of start and end of displacement [pixels]
    '''
    matches = _get_matches(descriptors1,
                           descriptors2, matcher, norm, verbose)
    x1, y1, x2, y2 = _filter_matches(matches, ratio_test,
                                     keyPoints1, keyPoints2, verbose)
    return x1, y1, x2, y2

def _get_matches(descriptors1, descriptors2, matcher, norm, verbose=False):
    ''' Match keypoints using BFMatcher with cv2.NORM_HAMMING '''
    t0 = time.time()
    bf = matcher(norm)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    t1 = time.time()
    if verbose:
        print('Keypoints matched', t1 - t0)
    return matches

def _filter_matches(matches, ratio_test, keyPoints1, keyPoints2, verbose=False):
    ''' Apply ratio test from Lowe '''
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    if verbose:
        print('Ratio test %f found %d keypoints' % (ratio_test, len(good)))

    # Coordinates for start, end point of vectors
    x1 = np.array([keyPoints1[m.queryIdx].pt[0] for m in good])
    y1 = np.array([keyPoints1[m.queryIdx].pt[1] for m in good])
    x2 = np.array([keyPoints2[m.trainIdx].pt[0] for m in good])
    y2 = np.array([keyPoints2[m.trainIdx].pt[1] for m in good])
    return x1, y1, x2, y2

def domain_filter(n, keyPoints, descr, domain, domainMargin=0, verbose=False, **kwargs):
    ''' Finds <keyPoints> from Nansat objects <n> which are within <domain>
    Parameters
    ----------
        n : source Nansat object
        keyPoints : list - keypoints on image from <n>
        descr : list - descriptors of <keyPoints>
        domain : destination Domain
        domainMargin : int - margin to crop points
    Returns
    -------
        keyPointsFilt : list of filtered keypoints
        descrFilt : list - descriptors of <keyPointsFilt>
    '''
    cols = [kp.pt[0] for kp in keyPoints]
    rows = [kp.pt[1] for kp in keyPoints]
    lon, lat = n.transform_points(cols, rows, 0)
    colsD, rowsD = domain.transform_points(lon, lat, 1)
    gpi = ((colsD >= 0 + domainMargin) *
           (rowsD >= 0 + domainMargin) *
           (colsD <= domain.shape()[1] - domainMargin) *
           (rowsD <= domain.shape()[0] - domainMargin))
    if verbose:
        print('Domain filter: %d -> %d' % (len(keyPoints), len(gpi[gpi])))
    return list(np.array(keyPoints)[gpi]), descr[gpi]

def max_drift_filter(n1, x1, y1, n2, x2, y2, max_speed=0.5, max_drift=None, verbose=False, **kwargs):
    ''' Filter out too high drift (m/s)
    Parameters
    ----------
        n1 : First Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        n2 : Second Nansat object
        x2 : 1D vector - X coordinates of keypoints on image 2
        y2 : 1D vector - Y coordinates of keypoints on image 2
        max_speed : float - maximum allowed ice drift speed, m/s
        max_drift : float - maximum allowed drift distance, meters

    Returns
    -------
        x1 : 1D vector - filtered source X coordinates on img1, pix
        y1 : 1D vector - filtered source Y coordinates on img1, pix
        x2 : 1D vector - filtered destination X coordinates on img2, pix
        y2 : 1D vector - filtered destination Y coordinates on img2, pix

    Note
    ----
        If time_coverage_start is not avaialabe from input data then the <max_speed> threshold
        is not used and the user should set value for <max_drift>.

    '''
    # chack if input datasets have time stamp
    try:
        n1_time_coverage_start = n1.time_coverage_start
        n2_time_coverage_start = n2.time_coverage_start
    except ValueError:
        data_has_timestamp = False
    else:
        data_has_timestamp = True

    if data_has_timestamp:
        # if datasets have timestamp compare with speed
        gpi = get_speed_ms(n1, x1, y1, n2, x2, y2) <= max_speed
    elif max_drift is not None:
        # if datasets don't have timestamp compare with displacement
        gpi = 1000.*get_displacement_km(n1, x1, y1, n2, x2, y2) <= max_drift
    else:
        # if max displacement is not given - raise error
        raise ValueError('''

        Error while filtering matching vectors!
        Input data does not have time stamp, and <max_drift> is not set.
        Either use data supported by Nansat, or
        provide a value for <max_drift> - maximum allowed ice displacement between images (meters).
        Examples:
            uft, vft, lon1ft, lat1ft, lon2ft, lat2ft = sid.get_drift_FT(max_drift=10000)
            x1, y1, x2, y2 = feature_tracking(n1, n2, max_drift=10000)
        Vectors with displacement higher than <max_drift> will be removed.

        ''')
    if verbose:
        print('MaxDrift filter: %d -> %d' % (len(x1), len(gpi[gpi])))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]

def lstsq_filter(x1, y1, x2, y2, psi=200, order=2, verbose=False, **kwargs):
    ''' Remove vectors that don't fit the model x1 = f(x2, y2)^n

    Fit the model x1 = f(x2, y2)^n using least squares method
    Simulate x1 using the model
    Compare actual and simulated x1 and remove points where error is too high
    Parameters
    ----------
        x1, y1, x2, y2 : coordinates of start and end of displacement [pixels]
        psi : threshold error between actual and simulated x1 [pixels]
    Returns
    -------
        x1 : 1D vector - filtered source X coordinates on img1, pix
        y1 : 1D vector - filtered source Y coordinates on img1, pix
        x2 : 1D vector - filtered destination X coordinates on img2, pix
        y2 : 1D vector - filtered destination Y coordinates on img2, pix
    '''
    if len(x1) == 0:
        return map(np.array, [[],[],[],[]])
    # interpolate using N-order polynomial
    x2sim, y2sim = interpolation_poly(x1, y1, x2, y2, x1, y1, order=order)

    # find error between actual and simulated x1
    err = np.hypot(x2 - x2sim, y2 - y2sim)

    # find pixels with error below psi
    gpi = err < psi

    if verbose:
        print('LSTSQ filter: %d -> %d' % (len(x1), len(gpi[gpi])))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]


def feature_tracking(n1, n2, **kwargs):
    ''' Run Feature Tracking Algrotihm on two images
    Parameters
    ----------
        n1 : First Nansat object with 2D UInt8 matrix
        n2 : Second Nansat object with 2D UInt8 matrix
        domainMargin : int - how much to crop from size of domain
        max_speed : float - maximum allow ice drift speed, m/s
        max_drift : float - maximum allow ice drift displacement, m
        **kwargs : parameters for functions:
            find_key_points
            domain_filter
            get_match_coords
            max_drift_filter
            lstsq_filter
    Returns
    -------
        x1 : 1D vector - source X coordinates on img1, pix
        y1 : 1D vector - source Y coordinates on img1, pix
        x2 : 1D vector - destination X coordinates on img2, pix
        y2 : 1D vector - destination Y coordinates on img2, pix
    '''
    # find many keypoints
    kp1, descr1 = find_key_points(n1[1], **kwargs)
    kp2, descr2 = find_key_points(n2[1], **kwargs)
    if len(kp1) < 2 or len(kp2) < 2:
        return (np.array([]),)*4

    # filter keypoints by Domain
    kp1, descr1 = domain_filter(n1, kp1, descr1, n2, **kwargs)
    if len(kp1) < 2:
        return (np.array([]),)*4
    kp2, descr2 = domain_filter(n2, kp2, descr2, n1, **kwargs)
    if len(kp2) < 2:
        return (np.array([]),)*4

    # find coordinates of matching key points
    x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2, **kwargs)

    # filter out pair with too high drift
    x1, y1, x2, y2 = max_drift_filter(n1, x1, y1, n2, x2, y2, **kwargs)

    # filter out inconsistent pairs
    x1, y1, x2, y2 = lstsq_filter(x1, y1, x2, y2, **kwargs)

    return x1, y1, x2, y2
