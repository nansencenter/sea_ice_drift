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

def get_uint8_image(image, vmin, vmax):
    ''' Scale image from float (or any) input array to uint8
    Parameters
    ----------
        image : 2D matrix
        vmin : float - minimum value
        vmax : float - maximum value
    Returns
    -------
        2D matrix
    '''
    # redistribute into range [0,255]
    uint8Image = 255 * (image - vmin) / (vmax - vmin)
    uint8Image[uint8Image < 0] = 0
    uint8Image[uint8Image > 255] = 255
    uint8Image[~np.isfinite(uint8Image)] = 0

    return uint8Image.astype('uint8')

def get_displacement_km(n1, x1, y1, n2, x2, y2, ll2km='domain'):
    ''' Find displacement in kilometers using Domain
    Parameters
    ----------
        n1 : First Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        n2 : Second Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
        ll2km : ['domain' or 'equirec'] - switch to compute distance
    Returns
    -------
        dx : 1D vector - eastward displacement, km
        dy : 1D vector - northward displacement, km
    '''
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
    ''' Find displacement in KM using Equirectangular approximation
    Parameters
    ----------
        lon1 : 1D vector - longitudes of keypoints on image 1
        lat1 : 1D vector - latitudes of keypoints on image 1
        lon2 : 1D vector - longitudes of keypoints on image 2
        lat2 : 1D vector - latitudes of keypoints on image 2
    Returns
    -------
        dx : 1D vector - eastward displacement, km
        dy : 1D vector - northward displacement, km
    '''
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
    ''' Find displacement in pixels of the first image
    Parameters
    ----------
        n1 : First Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        n2 : Second Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
    Returns
    -------
        dx : 1D vector - eastward displacement, pix
        dy : 1D vector - northward displacement, pix
    '''
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

def x2y2_interpolation_poly(x1, y1, x2, y2, x1grd, y1grd, order=1, **kwargs):
    ''' Interpolate values of x2/y2 onto full-res grids of x1/y1 using
    polynomial of order 1 (or 2 or 3)
    Parameters
    ----------
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
        x1grd : 1D vector - source X coordinate on img1
        y1grd : 1D vector - source Y coordinate on img2
        order : [1,2,3] - order of polynom
    Returns
    -------
        x2grd : 1D vector - destination X coordinate on img1
        y2grd : 1D vector - destination Y coordinate on img2
    '''
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

def x2y2_interpolation_near(x1, y1, x2, y2, x1grd, y1grd, method='linear', **kwargs):
    ''' Interpolate values of x2/y2 onto full-res grids of x1/y1 using
    linear interpolation of nearest points
    Parameters
    ----------
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
        x1grd : 1D vector - source X coordinate on img1
        y1grd : 1D vector - source Y coordinate on img2
        method : str - parameter for SciPy griddata
    Returns
    -------
        x2grd : 1D vector - destination X coordinate on img1
        y2grd : 1D vector - destination Y coordinate on img2
    '''
    src = np.array([y1, x1]).T
    dst = np.array([y1grd, x1grd]).T
    x2grd = griddata(src, x2, dst, method=method).T
    y2grd = griddata(src, y2, dst, method=method).T

    return x2grd, y2grd

def get_n(filename, bandName='sigma0_HV', factor=0.5,
                        vmin=-30, vmax=-5, denoise=False, dB=True,
                        **kwargs):
    ''' Get Nansat object with image data scaled to UInt8
    Parameters
    ----------
        filename : str - input file name
        bandName : str - name of band in the file
        factor : float - subsampling factor
        vmin : float - minimum allowed value in the band
        vmax : float - maximum allowed value in the band
        denoise : bool - apply denoising of sigma0 ?
        dB : bool - apply conversion to dB ?
        **kwargs : parameters for get_denoised_object()
    Returns
    -------
        n : Nansat object with one band scaled to UInt8
    '''
    if denoise:
        # run denoising
        n = get_denoised_object(filename, bandName, factor, **kwargs)
    else:
        # open data with Nansat and downsample
        n = Nansat(filename)
        n.resize(factor, eResampleAlg=-1)
    # improve geonetric accuracy
    n.reproject_GCPs()
    n.vrt.tps = True
    # get matrix with data
    img = n[bandName]
    # convert to dB
    if not denoise and dB:
        img = 10 * np.log10(img)
    # convert to 0 - 255
    img = get_uint8_image(img, vmin, vmax)

    nout = Nansat(domain=n, array=img, parameters={'name': bandName})
    return nout

def get_drift_vectors(n1, x1, y1, n2, x2, y2, ll2km='domain', **kwargs):
    ''' Find ice drift speed m/s
    Parameters
    ----------
        n1 : First Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        n2 : Second Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
        ll2km : ['domain' or 'equirec'] - switch to compute distance
    Returns
    -------
        u : 1D vector - eastward ice drift speed
        v : 1D vector - northward ice drift speed
        lon1 : 1D vector - longitudes of source points
        lat1 : 1D vector - latitudes of source points
        lon2 : 1D vector - longitudes of destination points
        lat2 : 1D vector - latitudes of destination points
    '''
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

def _fill_gpi(shape, gpi, data):
    ''' Fill 1D <data> into 2D matrix with <shape> based on 1D <gpi> '''
    y = np.zeros(shape).flatten()
    y[gpi] = data
    return y.reshape(shape)
