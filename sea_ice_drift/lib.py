# Name:    lib.py
# Purpose: Container of common functions
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

from scipy.interpolate import griddata

from nansat import Nansat, Domain

def reproject_gcp_to_stere(n):
    ''' Change projection of GCPs to stereographic add TPS option '''
    lon, lat = n.get_border()
    # reproject Ground Control Points (GCPS) to stereographic projection
    n.reproject_GCPs('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon.mean(), lat.mean()))
    n.vrt.tps = True

    return n

def get_uint8_image(image, vmin, vmax):
    ''' Scale image from float (or any) input array to uint8 '''
    # redistribute into range [0,255]
    uint8Image = 255 * (image - vmin) / (vmax - vmin)
    uint8Image[uint8Image < 0] = 0
    uint8Image[uint8Image > 255] = 255
    uint8Image[~np.isfinite(uint8Image)] = 0

    return uint8Image.astype('uint8')

def get_displacement_km(n1, x1, y1, n2, x2, y2, ll2km='domain'):
    ''' Find displacement in kilometers using Domain'''
    lon1, lat1 = n1.transform_points(x1, y1)
    lon2, lat2 = n2.transform_points(x2, y2)

    if ll2km == 'domain':
        d = Domain('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon1.mean(),
                                                                 lat1.mean()),
                    '-te -100000 -100000 100000 100000 -tr 1000 1000')
        x1d, y1d = d.transform_points(lon1, lat1, 1)
        x2d, y2d = d.transform_points(lon2, lat2, 1)
        dx = x2d - x1d
        dy = y1d - y2d
    elif ll2km == 'equirec':
        dx, dy = get_displacement_km_equirec(lon1, lat1, lon2, lat2)

    return dx, dy

def get_displacement_km_equirec(lon1, lat1, lon2, lat2):
    # U,V Equirectangular
    dlong = (lon2 - lon1)*np.pi/180;
    dlat  = (lat2 - lat1)*np.pi/180;
    slat  = (lat1 + lat2)*np.pi/180;
    p1 = (dlong)*np.cos(0.5*slat)
    p2 = (dlat)
    dx = 6371.000 * p1
    dy = 6371.000 * p2

    return dx, dy

def get_displacement_pix(n1, x1, y1, n2, x2, y2):
    ''' Find displacement in pixels of the first image'''
    lon2, lat2 = n2.transform_points(x2, y2)
    x2n1, y2n1 = n1.transform_points(lon2, lat2, 1)

    return x2n1 - x1, y2n1 - y1

def get_denoised_object(filename, bandName, factor, **kwargs):
    ''' Use sentinel1denoised and preform thermal noise removal
    Import is done within the function to make the dependency not so strict
    '''
    from sentinel1denoised.S1_EW_GRD_NoiseCorrection import Sentinel1Image
    s = Sentinel1Image(filename)
    s.add_denoised_band('sigma0_HV', **kwargs)
    s.resize(factor, eResampleAlg=-1)
    img = s[bandName + '_denoised']

    n = Nansat(domain=s)
    n.add_band(img, parameters=s.get_metadata(bandID=bandName))
    n.set_metadata(s.get_metadata())

    return n

def x2y2_interpolation_poly(x1, y1, x2, y2, x1grd, y1grd, order=1):
    ''' Interpolate values of x2/y2 onto full-res grids of x1/y1 using
    polynomial of order 1 (or 2 or 3)'''
    A = [np.ones(len(x1)), x1, y1]
    if order > 1:
        A += [x1**2, y1**2, x1*y1]
    if order > 2:
        A += [x1**3, y1**3, x1**2*y1, y1**2*x1]

    A = np.vstack(A).T
    Bx = np.linalg.lstsq(A, x2)[0]
    By = np.linalg.lstsq(A, y2)[0]
    x1grdF = x1grd.flatten()
    y1grdF = y1grd.flatten()

    A = [np.ones(len(x1grdF)), x1grdF, y1grdF]
    if order > 1:
        A += [x1grdF**2, y1grdF**2, x1grdF*y1grdF]
    if order > 2:
        A += [x1grdF**3, y1grdF**3, x1grdF**2*y1grdF, y1grdF**2*x1grdF]
    A = np.vstack(A).T
    x2grd = np.dot(A, Bx).reshape(x1grd.shape)
    y2grd = np.dot(A, By).reshape(x1grd.shape)

    return x2grd, y2grd

def x2y2_interpolation_near(x1, y1, x2, y2, x1grd, y1grd, method='linear'):
    ''' Interpolate values of x2/y2 onto full-res grids of x1/y1 using
    linear interpolation of nearest points '''
    src = np.array([y1, x1]).T
    dst = np.array([y1grd, x1grd]).T
    x2grd = griddata(src, x2, dst, method=method).T
    y2grd = griddata(src, y2, dst, method=method).T

    return x2grd, y2grd

def get_n_img(filename, bandName, factor, vmin, vmax, denoise, dB,
                                                              **kwargs):
    ''' Get Nansat object and matrix with scaled image data'''
    if denoise:
        # run denoising
        n = get_denoised_object(filename, bandName, factor, **kwargs)
    else:
        # open data with Nansat and downsample
        n = Nansat(filename)
        n.resize(factor, eResampleAlg=-1)
    # improve geonetric accuracy
    n = reproject_gcp_to_stere(n)
    n.vrt.tps = True
    # get matrix with data
    img = n[bandName]
    # convert to dB
    if not denoise and dB:
        img = 10 * np.log10(img)
    # convert to 0 - 255
    img = get_uint8_image(img, vmin, vmax)

    return n, img
