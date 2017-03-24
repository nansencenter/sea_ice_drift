# Name:    pmlib.py
# Purpose: Container of Pattern Matching functions
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

from multiprocessing import Pool

import numpy as np
from scipy import ndimage as nd

import cv2
import gdal

from sea_ice_drift.lib import (x2y2_interpolation_poly,
                               x2y2_interpolation_near,
                               get_drift_vectors,
                               _fill_gpi)

x1_dst_shared = None
y1_dst_shared = None
x2fg_shared = None
y2fg_shared = None
border_shared = None
img_size_shared = None
img1_shared = None
img2_shared = None
angles_shared = None
alpha0_shared = None


def get_rotated_template(img, r, c, size, angle, order=1):
    ''' Get rotated template of a given size
    Parameters
    ----------
        img : 2D numpy array - original image
        r : int - row coordinate of center
        c : int - column coordinate of center
        size : int - template size
        angle : float - rotation angle
        order : resampling order
    Returns
    -------
        templateRot : 2D numpy array - rotated subimage
    '''
    hws = size / 2.
    angle_rad = np.radians(angle)
    hwsrot = np.ceil(hws * np.abs(np.cos(angle_rad)) +
                     hws * np.abs(np.sin(angle_rad)))
    hwsrot2 = np.ceil(hwsrot * np.abs(np.cos(angle_rad)) +
                      hwsrot * np.abs(np.sin(angle_rad)))
    rotBorder1 = int(hwsrot2 - hws)
    rotBorder2 = int(rotBorder1 + hws + hws)

    # read large subimage
    if isinstance(img, np.ndarray):
        template = img[int(r-hwsrot):int(r+hwsrot+1), int(c-hwsrot):int(c+hwsrot+1)]
    elif isinstance(img, gdal.Dataset):
        template = img.ReadAsArray(xoff=int(c[0]-hwsrot),
                                   yoff=int(r[0]-hwsrot),
                                   xsize=int(hwsrot*2+1),
                                   ysize=int(hwsrot*2+1))
        
    templateRot = nd.interpolation.rotate(template, angle, order=order)
    templateRot = templateRot[rotBorder1:rotBorder2, rotBorder1:rotBorder2]

    return templateRot

def get_distance_to_nearest_keypoint(x1, y1, shape):
    ''' Return full-res matrix with distance to nearest keypoint in pixels
    Parameters
    ----------
        x1 : 1D vector - X coordinates of keypoints
        y1 : 1D vector - Y coordinates of keypoints
        shape : shape of image
    Returns
    -------
        dist : 2D numpy array - image with distances
    '''
    seed = np.zeros(shape, dtype=bool)
    seed[np.uint16(y1), np.uint16(x1)] = True
    dist = nd.distance_transform_edt(~seed,
                                    return_distances=True,
                                    return_indices=False)
    return dist

def get_initial_rotation(n1, n2):
    ''' Returns angle <alpha> of rotation between two Nansat <n1>, <n2>'''
    corners_n2_lons, corners_n2_lats = n2.get_corners()
    corner0_n2_x1, corner0_n2_y1 = n1.transform_points([corners_n2_lons[0]], [corners_n2_lats[0]], 1)
    corner1_n2_x1, corner1_n2_y1 = n1.transform_points([corners_n2_lons[1]], [corners_n2_lats[1]], 1)
    b = corner1_n2_x1 - corner0_n2_x1
    a = corner1_n2_y1 - corner0_n2_y1
    alpha = np.degrees(np.arctan2(b, a)[0])
    return alpha

def rotate_and_match(img1, x, y, img_size, image, alpha0, angles=[0],
                     mtype=cv2.TM_CCOEFF_NORMED):
    ''' Rotate template in a range of angles and run MCC for each
    Parameters
    ----------
        im1g : 2D numpy array - original image 1
        x : int - X coordinate of center
        y : int - Y coordinate of center
        img_size : size of template
        image : original image 2
        alpha0 : float - angle of rotation between two SAR scenes
        angles : list - which angles to test
        mtype : int - type of cross-correlation
    Returns
    -------
        best_r : float - MCC
        best_a : float - angle that gives highest MCC
        dx : int - X displacement of MCC
        dy : int - Y displacement of MCC
        best_result : 2D array - CC
        best_template : 2D array - template rotated to the best angle
    '''
    best_r = -np.inf
    for angle in angles:
        template = get_rotated_template(img1, y, x, img_size, angle-alpha0)
        if template.shape[0] < img_size or template.shape[1] < img_size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        result = cv2.matchTemplate(image, template.astype(np.uint8), mtype)
        ij = np.unravel_index(np.argmax(result), result.shape)
        if result.max() > best_r:
            best_r = result.max()
            best_a = angle
            best_result = result
            best_template = template
            best_ij = ij

    dy = best_ij[0] - (image.shape[0] - template.shape[0]) / 2.
    dx = best_ij[1] - (image.shape[1] - template.shape[1]) / 2.

    return best_r, best_a, dx, dy, best_result, best_template

def use_mcc(x1p, y1p, x2p, y2p, brd, img_size, img1, img2, alpha0, angles=[0]):
    ''' Apply MCC algorithm for one point
    Parameters
    ----------
        x1p : float, X coordinate on image 1
        y1p : float, Y coordinate on image 1
        x2p : float, first guess X coordinate on image 2
        y2p : float, first guess Y coordinate on image 2
        brd : int, searching distance (border around template)
        img_size : int, template size
        img1 : 2D array - full szie image 1
        img2 : 2D array - full szie image 2
        alpha0 : float, rotation between two images
        angles : list, which angles to test
    Returns
    -------
        x2 : float, result X coordinate on image 2
        y2 : float, result X coordinate on image 2
        r : float, MCC
        a : float, angle that gives highest MCC
    '''
    hws = int(img_size / 2.)
    image = img2[y2p-hws-brd:y2p+hws+brd+1,
                 x2p-hws-brd:x2p+hws+brd+1]
    #import ipdb; ipdb.set_trace()
    r,a,dx,dy,bestr,bestt = rotate_and_match(img1, x1p, y1p, img_size, image, alpha0, angles)

    x2 = x2p + dx
    y2 = y2p + dy

    return x2, y2, r, a

def use_mcc_mp(i):
    ''' Use MCC in multprocessing
    Uses global variables where first guess and images are stored
    Parameters
    ---------
        i : int, index of poin
    Returns
    -------
        x2 : float, result X coordinate on image 2
        y2 : float, result X coordinate on image 2
        r : float, MCC
        a : float, angle that gives highest MCC
    '''
    global x1_dst_shared, y1_dst_shared
    global x2fg_shared, y2fg_shared, border_shared
    global img_size_shared, img1_shared, img2_shared
    global angles_shared, alpha0_shared

    x2, y2, r, a = use_mcc(x1_dst_shared[i], y1_dst_shared[i],
                   x2fg_shared[i], y2fg_shared[i], border_shared[i],
                   img_size_shared,
                   img1_shared, img2_shared, alpha0_shared, angles_shared)
    if i % 10 == 0:
        print 100 * float(i) / len(x1_dst_shared), '%', x1_dst_shared[i], y1_dst_shared[i], x2, y2, r, a
    return x2, y2, r, a

def _init_pool(x1_dst, y1_dst, x2fg, y2fg, border, gpi, img_size,
              img1, img2, alpha0, angles):
    ''' Initialize data for multiprocessing '''
    global x1_dst_shared, y1_dst_shared
    global x2fg_shared, y2fg_shared, border_shared
    global img_size_shared, img1_shared, img2_shared
    global angles_shared, alpha0_shared
    x1_dst_shared = x1_dst[gpi]
    y1_dst_shared = y1_dst[gpi]
    x2fg_shared = x2fg[gpi]
    y2fg_shared = y2fg[gpi]
    border_shared = border[gpi]
    img_size_shared = img_size
    img1_shared = img1
    img2_shared = img2
    angles_shared = angles
    alpha0_shared = alpha0

def prepare_first_guess(x1_dst, y1_dst, x1, y1, x2, y2, img1, img_size,
                        min_border=20, max_border=50, **kwargs):
    ''' For the given coordinates estimate the First Guess
    Parameters
    ---------
        x1_dst : 1D vector, X coordinates of results on image 1
        y1_dst : 1D vector, Y coordinates of results on image 1
        x1 : 1D vector, X coordinates of keypoints on image 1
        y1 : 1D vector, Y coordinates of keypoints on image 1
        x2 : 1D vector, X coordinates of keypoints on image 2
        y2 : 1D vector, Y coordinates of keypoints on image 2
        img1 : 2D array, the fist image
        img_size : int, size of template
        min_border : int, minimum searching distance
        max_border : int, maximum searching distance
        **kwargs : parameters for:
            x2y2_interpolation_poly
            x2y2_interpolation_near
    Returns
    -------
        x2fg : 1D vector, first guess X coordinates of results on image 2
        y2fg : 1D vector, first guess X coordinates of results on image 2
        border : 1D vector, searching distance
    '''
    # interpolate 1st guess using 2nd order polynomial
    x2p2, y2p2 = x2y2_interpolation_poly(x1, y1, x2, y2, x1_dst, y1_dst, **kwargs)

    # interpolate 1st guess using griddata
    x2fg, y2fg = x2y2_interpolation_near(x1, y1, x2, y2, x1_dst, y1_dst, **kwargs)

    # find distance to nearest neigbour and create border matrix
    border_img = get_distance_to_nearest_keypoint(x1, y1, img1.shape)
    border = np.zeros(len(x1_dst)) + max_border
    gpi = ((x1_dst >= 0) * (x1_dst < img1.shape[1]) *
           (y1_dst >= 0) * (y1_dst < img1.shape[0]))
    border[gpi] = border_img[y1_dst.astype(np.int16)[gpi],
                             x1_dst.astype(np.int16)[gpi]]

    # define searching distance
    border[border < min_border] = min_border
    border[border > max_border] = max_border
    border[np.isnan(y2fg)] = max_border

    # define FG based on P2 and GD
    x2fg[np.isnan(x2fg)] = x2p2[np.isnan(x2fg)]
    y2fg[np.isnan(y2fg)] = y2p2[np.isnan(y2fg)]

    return x2fg, y2fg, border

def pattern_matching(lon1_dst, lat1_dst,
                     n1, x1, y1, n2, x2, y2,
                     img_size=35, threads=5, angles=range(-15,16,3),
                     **kwargs):
    ''' Run Pattern Matching Algorithm on two images
    Parameters
    ---------
        lon_dst : 1D vector, longitude of results on image 1
        lon_dst : 1D vector, latitude of results on image 1
        n1 : Nansat, the fist image with 2D array        
        x1 : 1D vector, X coordinates of keypoints on image 1
        y1 : 1D vector, Y coordinates of keypoints on image 1
        n2 : Nansat, the second image with 2D array        
        x2 : 1D vector, X coordinates of keypoints on image 2
        y2 : 1D vector, Y coordinates of keypoints on image 2
        img_size : int, size of template
        threads : int, number of parallel threads
        angles : 1D vector, angles for template rotation
        **kwargs : parameters for:
            prepare_first_guess
            get_drift_vectors
    Returns
    -------
        u : 1D vector, eastward ice drift speed, m/s
        v : 1D vector, eastward ice drift speed, m/s
        r : 1D vector, MCC
        a : 1D vector, angle that gives the highes MCC
        lon2_dst : 1D vector, longitude of results on image 2
        lat2_dst : 1D vector, latitude  of results on image 2
    '''    
    # convert lon/lat to pixe/line of the first image
    x1_dst, y1_dst = n1.transform_points(lon1_dst.flatten(), lat1_dst.flatten(), 1)

    x2fg, y2fg, border = prepare_first_guess(x1_dst, y1_dst,
                                             x1, y1, x2, y2, n1[1],
                                             img_size,
                                             **kwargs)
    # find good input points
    hws = img_size / 2
    hws_hypot = np.hypot(hws, hws)
    gpi = ((x2fg-border-hws > 0) * (x2fg+border+hws < n2.shape()[1]) *
           (y2fg-border-hws > 0) * (y2fg+border+hws < n2.shape()[0]) *
           (x1_dst-hws_hypot > 0) * (x1_dst+hws_hypot < n1.shape()[1]) *
           (y1_dst-hws_hypot > 0) * (y1_dst+hws_hypot < n1.shape()[1]))

    alpha0 = get_initial_rotation(n1, n2)

    # run MCC in multiple threads
    p = Pool(threads, initializer=_init_pool,
            initargs=(x1_dst, y1_dst, x2fg, y2fg, border, gpi,
            img_size, n1[1], n2[1], alpha0, angles))
    results = p.map(use_mcc_mp, range(len(gpi[gpi])))
    
    x2_dst = np.array(results)[:,0]
    y2_dst = np.array(results)[:,1]
    r = np.array(results)[:,2]
    a = np.array(results)[:,3]

    u, v, lon1, lat1, lon2, lat2 = get_drift_vectors(n1, x1_dst[gpi], y1_dst[gpi],
                                                     n2, x2_dst, y2_dst,
                                                     **kwargs)

    lon2_dst = _fill_gpi(lon1_dst.shape, gpi, lon2)
    lat2_dst = _fill_gpi(lon1_dst.shape, gpi, lat2)
    u = _fill_gpi(lon1_dst.shape, gpi, u)
    v = _fill_gpi(lon1_dst.shape, gpi, v)
    r = _fill_gpi(lon1_dst.shape, gpi, r)
    a = _fill_gpi(lon1_dst.shape, gpi, a)
    
    return u, v, r, a, lon2_dst, lat2_dst

    
