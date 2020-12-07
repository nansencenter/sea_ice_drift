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
from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import zoom, maximum_filter
from scipy.interpolate import griddata
from osgeo import gdal

from nansat import Nansat, Domain, NSR

AVG_EARTH_RADIUS = 6371  # in km

def get_uint8_image(image, vmin, vmax, pmin, pmax):
    ''' Scale image from float (or any) input array to uint8
    Parameters
    ----------
    image : 2D ndarray
        matrix with sigma0 image
    vmin : float or None
        minimum value to convert to 1
    vmax : float or None
        maximum value to convert to 255
    pmin : float
        lower percentile for data scaling if vmin is None
    pmax : float
        upper percentile for data scaling if vmax is None

    Returns
    -------
        2D matrix
    '''
    if vmin is None:
        vmin = np.nanpercentile(image, pmin)
        print('VMIN: ', vmin)
    if vmax is None:
        vmax = np.nanpercentile(image, pmax)
        print('VMAX: ', vmax)
    # redistribute into range [1,255]
    # 0 is reserved for invalid pixels
    uint8Image = 1 + 254 * (image - vmin) / (vmax - vmin)
    uint8Image[uint8Image < 1] = 1
    uint8Image[uint8Image > 255] = 255
    uint8Image[~np.isfinite(image)] = 0

    return uint8Image.astype('uint8')

def get_displacement_km(n1, x1, y1, n2, x2, y2):
    ''' Find displacement in kilometers using Haversine
        http://www.movable-type.co.uk/scripts/latlong.html
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
        h : 1D vector - total displacement, km
    '''
    lon1, lat1 = n1.transform_points(x1, y1)
    lon2, lat2 = n2.transform_points(x2, y2)

    lt1, ln1, lt2, ln2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lt2 - lt1
    dlon = ln2 - ln1
    d = (np.sin(dlat * 0.5) ** 2 +
         np.cos(lt1) * np.cos(lt2) * np.sin(dlon * 0.5) ** 2)
    return 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

def get_speed_ms(n1, x1, y1, n2, x2, y2):
    ''' Find ice drift speed in m/s
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
        spd : 1D vector - speed, m/s
    '''
    dt = (n2.time_coverage_start - n1.time_coverage_start).total_seconds()
    return 1000.*get_displacement_km(n1, x1, y1, n2, x2, y2)/abs(dt)

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
        dx : 1D vector - leftward displacement, pix
        dy : 1D vector - upward displacement, pix
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

def interpolation_poly(x1, y1, x2, y2, x1grd, y1grd, order=1, **kwargs):
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
    Bx = np.linalg.lstsq(A, x2, rcond=-1)[0]
    By = np.linalg.lstsq(A, y2, rcond=-1)[0]
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

def interpolation_near(x1, y1, x2, y2, x1grd, y1grd, method='linear', **kwargs):
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

def hh_angular_correction(n, img, bandName, correct_hh_factor):
    """ Correct sigma0_HH for incidence angle dependence

    Paramaters
    ----------
    correct_hh_factor : float
        coefficient in the correction factor sigma0_HH_cor = sigma0_HH + correct_hh_factor * incidence_angle

    Returns
    -------
    img : ndarray
        corrected sigma0_HH in dB

    """
    if bandName == 'sigma0_HH' and n.has_band('incidence_angle'):
        ia = n['incidence_angle']
        imgcor = img - ia * correct_hh_factor
    else:
        imgcor = img

    return imgcor

def get_spatial_mean(img):
    """ Approximate spatial mean brightness by second order polynomial

    Paramaters
    ----------
    img : 2D ndimage
        input image

    Returns
    -------
    img2 : ndarray
        approximated mean brightness

    """
    step =50
    cols, rows = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    imgsub, rowsub, colsub = [v[::step, ::step] for v in [img, rows, cols]]
    gpi = np.isfinite(imgsub) * (imgsub > np.nanpercentile(imgsub, 5))
    imgsub, rowsub, colsub = [v[gpi] for v in [imgsub, rowsub, colsub]]
    def get_predictor(x, y):
        return np.array([x, x**2, y, y**2, x*y, np.ones_like(x)]).T
    A = get_predictor(colsub, rowsub)
    x = np.linalg.lstsq(A, imgsub, rcond=None)
    img2  = x[0][0] * cols
    img2 += x[0][1] * cols**2
    img2 += x[0][2] * rows
    img2 += x[0][3] * rows**2
    img2 += x[0][4] * cols*rows
    img2 += x[0][5]
    return img2

def get_n(filename, bandName='sigma0_HV',
                    factor=0.5,
                    denoise=False,
                    dB=True,
                    mask_invalid=True,
                    landmask_border=20,
                    correct_hh=False,
                    correct_hh_factor=-0.27,
                    remove_spatial_mean=False,
                    vmin=None,
                    vmax=None,
                    pmin=10,
                    pmax=99,
                    **kwargs):
    """ Get Nansat object with image data scaled to UInt8
    Parameters
    ----------
    filename : str
        input file name
    bandName : str
        name of band in the file
    factor : float
        subsampling factor
    denoise : bool
        apply denoising of sigma0 ?
    dB : bool
        apply conversion to dB ?
    mask_invalid : bool
        mask invalid pixels (land, inf, etc) with 0 ?
    landmask_border : int
        border around landmask
    correct_hh : bool
        perform angular correction of sigma0_HH ?
    correct_hh_factor : float
        coefficient in the correction factor sigma0_HH_cor = sigma0_HH + correct_hh_factor * incidence_angle
    remove_spatial_mean : bool
        remove spatial mean from image ?
    vmin : float or None
        minimum value to convert to 1
    vmax : float or None
        maximum value to convert to 255
    pmin : float
        lower percentile for data scaling if vmin is None
    pmax : float
        upper percentile for data scaling if vmax is None
    **kwargs : dummy parameters for
        get_denoised_object()

    Returns
    -------
        n : Nansat object with one band scaled to UInt8

    """
    if denoise:
        # run denoising
        n = get_denoised_object(filename, bandName, factor, **kwargs)
    else:
        # open data with Nansat and downsample
        n = Nansat(filename)
        if factor != 1:
            n.resize(factor, resample_alg=-1)
    # get matrix with data
    img = n[bandName]
    # convert to dB
    if not denoise and dB:
        img[img <= 0] = np.nan
        img = 10 * np.log10(img)
    if correct_hh:
        img = hh_angular_correction(n, img, bandName, correct_hh_factor)
    if mask_invalid:
        mask = get_invalid_mask(img, n, landmask_border)
        img[mask] = np.nan
    if remove_spatial_mean:
        img -= get_spatial_mean(img)
    # convert to 0 - 255
    img = get_uint8_image(img, vmin, vmax, pmin, pmax)
    # create Nansat with one band only
    nout = Nansat.from_domain(n, img, parameters={'name': bandName})
    nout.set_metadata(n.get_metadata())
    # improve geolocation accuracy
    if len(nout.vrt.dataset.GetGCPs()) > 0:
        nout.reproject_gcps()
        nout.vrt.tps = True

    return nout

def get_invalid_mask(img, n, landmask_border):
    """
    Create mask of invalid pixels (land, cosatal, inf)

    Parameters
    ----------
    img : float ndarray
        input image
    n : Nansat
        input Nansat object
    landmask_border : int
        border around landmask

    Returns
    -------
    mask : 2D bool ndarray
        True where pixels are invalid
    """
    mask = np.isnan(img) + np.isinf(img)
    n.resize(1./landmask_border)
    try:
        wm = n.watermask()[1]
    except:
        print('Cannot add landmask')
    else:
        wm[wm > 2] = 2
        wmf = maximum_filter(wm, 3)
        wmz = zoom(wmf, (np.array(img.shape) / np.array(wm.shape)))
        mask[wmz == 2] = True

    n.undo()
    return mask

def get_drift_vectors(n1, x1, y1, n2, x2, y2, nsr=NSR(), **kwargs):
    ''' Find ice drift speed m/s
    Parameters
    ----------
        n1 : First Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 1
        y1 : 1D vector - Y coordinates of keypoints on image 1
        n2 : Second Nansat object
        x1 : 1D vector - X coordinates of keypoints on image 2
        y1 : 1D vector - Y coordinates of keypoints on image 2
        nsr: Nansat.NSR(), projection that defines the grid
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

    # create domain that converts lon/lat to units of the projection
    d = Domain(nsr, '-te -10 -10 10 10 -tr 1 1')

    # find displacement in needed units
    x1, y1 = d.transform_points(lon1, lat1, 1)
    x2, y2 = d.transform_points(lon2, lat2, 1)

    return x2-x1, y1-y2, lon1, lat1, lon2, lat2

def _fill_gpi(shape, gpi, data):
    ''' Fill 1D <data> into 2D matrix with <shape> based on 1D <gpi> '''
    y = np.zeros(shape).flatten() + np.nan
    y[gpi] = data
    return y.reshape(shape)
