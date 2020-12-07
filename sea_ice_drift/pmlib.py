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
from __future__ import absolute_import, print_function

import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2

from nansat import NSR

from sea_ice_drift.lib import (interpolation_poly,
                               interpolation_near,
                               get_drift_vectors,
                               _fill_gpi)

shared_args = None
shared_kwargs = None

def get_hessian(ccm, hes_norm=True, hes_smth=False, **kwargs):
    """ Find Hessian of the input cross correlation matrix <ccm>

    Parameters
    ----------
    ccm : 2D numpy array, cross-correlation matrix
    hes_norm : bool, normalize Hessian by AVG and STD?
    hes_smth : bool, smooth Hessian?

    """
    if hes_smth:
        ccm2 = nd.filters.gaussian_filter(ccm, 1)
    else:
        ccm2 = ccm
    # Jacobian components
    dcc_dy, dcc_dx = np.gradient(ccm2)
    # Hessian components
    d2cc_dx2 = np.gradient(dcc_dx)[1]
    d2cc_dy2 = np.gradient(dcc_dy)[0]
    hes = np.hypot(d2cc_dx2, d2cc_dy2)
    if hes_norm:
        hes = (hes - np.median(hes)) / np.std(hes)

    return hes

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

def get_template(img, c, r, a, s, rot_order=0, **kwargs):
    """ Get rotated and shifted square template
    Parameters
    ----------
        img : ndarray, input image
        c : float, center column coordinate (pixels)
        r : float, center row coordinate (pixels)
        a : float, rotation angle (degrees)
        s : odd int, template size (width and height)
        order : int, transformation order
    Returns
    -------
        t : ndarray (s,s)[np.uint8], rotated template

    """
    # center on output template
    tc = int(s / 2.) + 1
    tc = np.array([tc, tc])

    a = np.radians(a)
    transform = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
    offset = np.array([r, c]) - tc.dot(transform)

    t = nd.interpolation.affine_transform(
        img, transform.T, order=rot_order, offset=offset, output_shape=(s, s), cval=0.0, output=np.uint8)

    return t

def rotate_and_match(img1, c1, r1, img_size, image2, alpha0,
                     angles=[-3,0,3],
                     mtype=cv2.TM_CCOEFF_NORMED,
                     template_matcher=cv2.matchTemplate,
                     mcc_norm=False,
                     **kwargs):
    ''' Rotate template in a range of angles and run MCC for each
    Parameters
    ----------
        im1 : 2D numpy array - original image 1
        c1 : float - column coordinate of center on img1
        r1 : float - row coordinate of center on img1
        img_size : size of template
        image : np.uint8, subset from image 2
        alpha0 : float - angle of rotation between two SAR scenes
        angles : list - which angles to test
        mtype : int - type of cross-correlation
        template_matcher : func - function to use for template matching
        mcc_norm : bool, normalize MCC by AVG and STD ?
        kwargs : dict, params for get_hessian
    Returns
    -------
        dc : int - column displacement of MCC
        dr : int - row displacement of MCC
        best_a : float - angle of MCC
        best_r : float - MCC
        best_h : float - Hessian at highest MCC point
        best_result : float ndarray - cross correlation matrix
        best_template : uint8 ndarray - best rotated template

    '''
    res_shape = [image2.shape[0] - img_size +1]*2
    best_r = -np.inf
    for angle in angles:
        template = get_template(img1, c1, r1, angle-alpha0, img_size, **kwargs)
        if ((template.min() == 0) or
            (template.shape[0] < img_size or template.shape[1] < img_size)):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        result = template_matcher(image2, template, mtype)

        ij = np.unravel_index(np.argmax(result), result.shape)

        if result.max() > best_r:
            best_r = result.max()
            best_a = angle
            best_result = result
            best_template = template
            best_ij = ij

    best_h = get_hessian(best_result, **kwargs)[best_ij]
    dr = best_ij[0] - (image2.shape[0] - template.shape[0]) / 2.
    dc = best_ij[1] - (image2.shape[1] - template.shape[1]) / 2.

    if mcc_norm:
        best_r = (best_r - np.median(best_result)) / np.std(best_result)

    return dc, dr, best_a, best_r, best_h, best_result, best_template

def use_mcc(c1, r1, c2fg, r2fg, border, img1, img2, img_size, alpha0, **kwargs):
    """ Apply MCC algorithm for one point

    Parameters
    ----------
        c1 : float, column coordinate on image 1
        r1 : float, row coordinate on image 1
        c2fg : int, first guess column coordinate on image 2
        r2fg : int, first guess row coordinate on image 2
        border : int, searching distance (border around template)
        img1 : 2D array - full szie image 1
        img2 : 2D array - full szie image 2
        img_size : int, template size
        alpha0 : float, rotation between two images
        kwargs : dict, params for rotate_and_match, get_template, get_hessian
    Returns
    -------
        x2 : float, result X coordinate on image 2
        y2 : float, result X coordinate on image 2
        a : float, angle that gives highest MCC
        r : float, MCC
        h : float, Hessian of CC at MCC point

    """
    hws = int(img_size / 2.)
    image = img2[int(r2fg-hws-border):int(r2fg+hws+border+1),
                 int(c2fg-hws-border):int(c2fg+hws+border+1)]

    dc, dr, best_a, best_r, best_h, best_result, best_template = rotate_and_match(img1, c1, r1,
                                                      img_size,
                                                      image,
                                                      alpha0,
                                                      **kwargs)
    c2 = c2fg + dc
    r2 = r2fg + dr

    return c2, r2, best_a, best_r, best_h

def use_mcc_mp(i):
    """ Use MCC
    Uses global variables where first guess and images are stored
    Parameters
    ---------
        i : int, index of point
    Returns
    -------
        c2 : float, result X coordinate on image 2
        r2 : float, result X coordinate on image 2
        a : float, angle that gives highest MCC
        r : float, MCC
        h : float, Hessian of CC at MCC point

    """
    global shared_args, shared_kwargs

    # structure of shared_args:
    # c1pm1i, r1pm1i, c2fg, r2fg, brd2, img1, img2, img_size, alpha0, kwargs
    c2, r2, a, r, h = use_mcc(shared_args[0][i],
                              shared_args[1][i],
                              shared_args[2][i],
                              shared_args[3][i],
                              shared_args[4][i],
                              shared_args[5],
                              shared_args[6],
                              shared_args[7],
                              shared_args[8],
                              **shared_kwargs)
    if i % 100 == 0:
        print('%02.0f%% %07.1f %07.1f %07.1f %07.1f %+05.1f %04.2f %04.2f' % (
            100 * float(i) / len(shared_args[0]),
            shared_args[0][i], shared_args[1][i], c2, r2, a, r, h), end='\r')
    return c2, r2, a, r, h

def prepare_first_guess(c2pm1, r2pm1, n1, c1, r1, n2, c2, r2, img_size,
                        min_fg_pts=5,
                        min_border=20,
                        max_border=50,
                        old_border=True, **kwargs):
    ''' For the given intial coordinates estimate the approximate final coordinates
    Parameters
    ---------
        c2_pm1 : 1D vector, initial PM column on image 2
        r2_pm1 : 1D vector, initial PM rows of image 2
        n1 : Nansat, the fist image with 2D array
        c1 : 1D vector, initial FT columns on img1
        r1 : 1D vector, initial FT rows on img2
        n2 : Nansat, the second image with 2D array
        c2 : 1D vector, final FT columns on img2
        r2 : 1D vector, final FT rows on img2
        img_size : int, size of template
        min_fg_pts : int, minimum number of fist guess points
        min_border : int, minimum searching distance
        max_border : int, maximum searching distance
        old_border : bool, use old border selection algorithm?
        **kwargs : parameters for:
            x2y2_interpolation_poly
            x2y2_interpolation_near
    Returns
    -------
        c2_fg : 1D vector, approximate final PM columns on img2 (first guess)
        r2_fg : 1D vector, approximate final PM rows on img2 (first guess)
        border : 1D vector, searching distance
    '''
    n2_shape = n2.shape()
    # convert initial FT points to coordinates on image 2
    lon1, lat1 = n1.transform_points(c1, r1)
    c1n2, r1n2 = n2.transform_points(lon1, lat1, 1)

    # interpolate 1st guess using 2nd order polynomial
    c2p2, r2p2 = np.round(interpolation_poly(c1n2, r1n2, c2, r2, c2pm1, r2pm1, **kwargs))

    # interpolate 1st guess using griddata
    c2fg, r2fg = np.round(interpolation_near(c1n2, r1n2, c2, r2, c2pm1, r2pm1, **kwargs))

    # TODO:
    # Now border is proportional to the distance to the point
    # BUT it assumes that:
    #     close to any point error is small, and
    #     error varies between points
    # What if error does not vary with distance from the point?
    # Border can be estimated as error of the first guess
    # (x2 - x2_predicted_with_polynom) gridded using nearest neighbour.
    if old_border:
        # find distance to nearest neigbour and create border matrix
        border_img = get_distance_to_nearest_keypoint(c2, r2, n2_shape)
        border = np.zeros(c2pm1.size) + max_border
        gpi = ((c2pm1 >= 0) * (c2pm1 < n2_shape[1]) *
               (r2pm1 >= 0) * (r2pm1 < n2_shape[0]))
        border[gpi] = border_img[np.round(r2pm1[gpi]).astype(np.int16),
                                 np.round(c2pm1[gpi]).astype(np.int16)]
    else:
        c2tst, r2tst = interpolation_poly(c1n2, r1n2, c2, r2, c1n2, r1n2, **kwargs)
        c2dif, r2dif = interpolation_near(c1n2, r1n2,
                                               c2-c2tst, r2-r2tst,
                                               c2pm1, r2pm1,
                                               **kwargs)
        border = np.hypot(c2dif, r2dif)

    # define searching distance
    border[border < min_border] = min_border
    border[border > max_border] = max_border
    border[np.isnan(c2fg)] = max_border
    border = np.floor(border)

    # define FG based on P2 and GD
    c2fg[np.isnan(c2fg)] = c2p2[np.isnan(c2fg)]
    r2fg[np.isnan(r2fg)] = r2p2[np.isnan(r2fg)]

    return c2fg, r2fg, border

def pattern_matching(lon_pm1, lat_pm1,
                     n1, c1, r1, n2, c2, r2,
                     margin=0,
                     img_size=35,
                     threads=5,
                     srs='+proj=latlong +datum=WGS84 +ellps=WGS84 +no_defs',
                     **kwargs):
    ''' Run Pattern Matching Algorithm on two images
    Parameters
    ---------
        lon_pm1 : 1D vector
            longitudes of destination initial points
        lat_pm1 : 1D vector
            latitudes of destination initial points
        n1 : Nansat
            the fist image with 2D array
        c1 : 1D vector
            initial FT columns on img1
        r1 : 1D vector
            initial FT rows on img2
        n2 : Nansat
            the second image with 2D array
        c2 : 1D vector
            final FT columns on img2
        r2 : 1D vector
            final FT rows on img2
        img_size : int
            size of template
        threads : int
            number of parallel threads
        srs: str
            destination spatial refernce system of the drift vectors (proj4 or WKT)
        **kwargs : optional parameters for:
            prepare_first_guess
                min_fg_pts : int, minimum number of fist guess points
                min_border : int, minimum searching distance
                max_border : int, maximum searching distance
                old_border : bool, use old border selection algorithm?
            rotate_and_match
                angles : list - which angles to test
                mtype : int - type of cross-correlation
                template_matcher : func - function to use for template matching
                mcc_norm : bool, normalize MCC by AVG and STD ?
            get_template
                rot_order : resampling order for rotation
            get_hessian
                hes_norm : bool, normalize Hessian by AVG and STD?
                hes_smth : bool, smooth Hessian?
            get_drift_vectors
                nsr: Nansat.NSR(), projection that defines the grid
    Returns
    -------
        u : 1D vector
            eastward ice drift displacement [destination SRS units]
        v : 1D vector
            northward ice drift displacement [destination SRS units]
        a : 1D vector
            angle that gives the highes MCC
        r : 1D vector
            Maximum cross correlation (MCC)
        h : 1D vector
            Hessian of CC at MCC point
        lon2_dst : 1D vector
            longitude of results on image 2
        lat2_dst : 1D vector
            latitude  of results on image 2
    '''
    t0 = time.time()
    img1, img2 = n1[1], n2[1]
    dst_shape = lon_pm1.shape

    # coordinates of starting PM points on image 2
    c2pm1, r2pm1 = n2.transform_points(lon_pm1.flatten(), lat_pm1.flatten(), 1)

    # integer coordinates of starting PM points on image 2
    c2pm1i, r2pm1i = np.round([c2pm1, r2pm1])

    # fake cooridinates for debugging
    #c2pm1, r2pm1 = np.meshgrid(np.arange(c2pm1i.min(), c2pm1i.max(), 25),
    #                           np.arange(c2pm1i.min(), c2pm1i.max(), 25))
    #dst_shape = c2pm1.shape
    #c2pm1i, r2pm1i = np.round([c2pm1.flatten(), r2pm1.flatten()])

    # coordinates of starting PM points on image 1 (correposond to integer coordinates in img2)
    lon1i, lat1i = n2.transform_points(c2pm1i, r2pm1i)
    c1pm1i, r1pm1i = n1.transform_points(lon1i, lat1i, 1)

    # approximate final PM points on image 2 (the first guess)
    c2fg, r2fg, brd2 = prepare_first_guess(c2pm1i, r2pm1i, n1, c1, r1, n2, c2, r2, img_size, **kwargs)

    # find valid input points
    hws = round(img_size / 2) + 1
    hws_hypot = np.hypot(hws, hws)
    gpi = ((c2fg-brd2-hws-margin > 0) *
           (r2fg-brd2-hws-margin > 0) *
           (c2fg+brd2+hws+margin < n2.shape()[1]) *
           (r2fg+brd2+hws+margin < n2.shape()[0]) *
           (c1pm1i-hws_hypot-margin > 0) *
           (r1pm1i-hws_hypot-margin > 0) *
           (c1pm1i+hws_hypot+margin < n1.shape()[1]) *
           (r1pm1i+hws_hypot+margin < n1.shape()[0]))

    alpha0 = get_initial_rotation(n1, n2)

    def _init_pool(*args):
        """ Initialize shared data for multiprocessing """
        global shared_args, shared_kwargs
        shared_args = args[:9]
        shared_kwargs = args[9]

    if threads <= 1:
        # run MCC without threads
        _init_pool(c1pm1i[gpi], r1pm1i[gpi], c2fg[gpi], r2fg[gpi], brd2[gpi], img1, img2, img_size, alpha0, kwargs)
        results = [use_mcc_mp(i) for i in range(len(gpi[gpi]))]
    else:
        # run MCC in multiple threads
        p = Pool(threads, initializer=_init_pool,
                initargs=(c1pm1i[gpi], r1pm1i[gpi], c2fg[gpi], r2fg[gpi], brd2[gpi], img1, img2, img_size, alpha0, kwargs))
        results = p.map(use_mcc_mp, range(len(gpi[gpi])))
        p.close()
        p.terminate()
        p.join()
        del p

    print('\n', 'Pattern matching - OK! (%3.0f sec)' % (time.time() - t0))
    if len(results) == 0:
        lon2_dst = np.zeros(dst_shape) + np.nan
        lat2_dst = np.zeros(dst_shape) + np.nan
        u = np.zeros(dst_shape) + np.nan
        v = np.zeros(dst_shape) + np.nan
        a = np.zeros(dst_shape) + np.nan
        r = np.zeros(dst_shape) + np.nan
        h = np.zeros(dst_shape) + np.nan
        lon_pm2_grd = np.zeros(dst_shape) + np.nan
        lat_pm2_grd = np.zeros(dst_shape) + np.nan
    else:
        results = np.array(results)

        # coordinates of final PM points on image 2 (correspond to integer intial coordinates)
        c2pm2i = results[:,0]
        r2pm2i = results[:,1]

        # coordinatesof final PM points on image 2 (correspond to real intial coordinates)
        dci, dri, = c2pm1 - c2pm1i,  r2pm1 - r2pm1i
        c2pm2, r2pm2 = c2pm2i + dci[gpi], r2pm2i + dri[gpi]

        # coordinates of initial PM points on destination grid and coordinates system
        xpm1, ypm1 = n2.transform_points(c2pm1, r2pm1, 0, NSR(srs))
        xpm1_grd = xpm1.reshape(dst_shape)
        ypm1_grd = ypm1.reshape(dst_shape)

        # coordinates of final PM points on destination grid and coordinates system
        xpm2, ypm2 = n2.transform_points(c2pm2, r2pm2, 0, NSR(srs))
        xpm2_grd = _fill_gpi(dst_shape, gpi, xpm2)
        ypm2_grd = _fill_gpi(dst_shape, gpi, ypm2)
        lon_pm2, lat_pm2 = n2.transform_points(c2pm2, r2pm2, 0)
        lon_pm2_grd = _fill_gpi(dst_shape, gpi, lon_pm2)
        lat_pm2_grd = _fill_gpi(dst_shape, gpi, lat_pm2)

        # speed vectors on destination grid and coordinates system
        u = xpm2_grd - xpm1_grd
        v = ypm2_grd - ypm1_grd

        # angle, correlation and hessian on destination grid
        a = results[:,2]
        r = results[:,3]
        h = results[:,4]
        a = _fill_gpi(dst_shape, gpi, a)
        r = _fill_gpi(dst_shape, gpi, r)
        h = _fill_gpi(dst_shape, gpi, h)

    return u, v, a, r, h, lon_pm2_grd, lat_pm2_grd
