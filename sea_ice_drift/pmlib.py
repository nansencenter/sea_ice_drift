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

import numpy as np

from scipy import ndimage as nd

import cv2

from nansat import Nansat, Domain

def get_rotated_template(img, r, c, size, angle, order=1):
    ''' Get rotated template of a given size '''
    hws = size / 2.
    angle_rad = np.radians(angle)
    hwsrot = np.ceil(hws * np.abs(np.cos(angle_rad)) +
                     hws * np.abs(np.sin(angle_rad)))
    hwsrot2 = np.ceil(hwsrot * np.abs(np.cos(angle_rad)) +
                      hwsrot * np.abs(np.sin(angle_rad)))
    rotBorder1 = hwsrot2 - hws
    rotBorder2 = rotBorder1 + hws + hws

    template = img[r-hwsrot:r+hwsrot+1, c-hwsrot:c+hwsrot+1]
    templateRot = nd.interpolation.rotate(template, angle, order=1)
    templateRot = templateRot[rotBorder1:rotBorder2, rotBorder1:rotBorder2]

    return templateRot

def get_distance_to_nearest_keypoint(x1, y1, shape):
    ''' Return full-res matrix with distance to nearest keypoint in pixels '''
    seed = np.zeros(shape, dtype=bool)
    seed[np.uint16(y1), np.uint16(x1)] = True
    dist = nd.distance_transform_edt(~seed,
                                    return_distances=True,
                                    return_indices=False)
    return dist

def get_initial_rotation(n1, n2):
    ''' Calculate angle of rotation between two images'''
    corners_n2_lons, corners_n2_lats = n2.get_corners()
    corner0_n2_x1, corner0_n2_y1 = n1.transform_points([corners_n2_lons[0]], [corners_n2_lats[0]], 1)
    corner1_n2_x1, corner1_n2_y1 = n1.transform_points([corners_n2_lons[1]], [corners_n2_lats[1]], 1)
    b = corner1_n2_x1 - corner0_n2_x1
    a = corner1_n2_y1 - corner0_n2_y1
    alpha = np.degrees(np.arctan2(b, a)[0])
    return alpha

def rotate_and_match(img1, x, y, img_size, image, alpha0=0, angles=[0],
                     mtype=cv2.TM_CCOEFF_NORMED):
    ''' Rotate template in a range of angles and run MCC for each '''
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

def use_mcc(x1, y1, x2grd_fg, y2grd_fg, border, img_size, img1, img2, angles=[0], alpha0=0):
    ''' Apply MCC algorithm for one point '''
    ix2 = x2grd_fg[y1, x1]
    iy2 = y2grd_fg[y1, x1]
    brd = border[y1, x1]
    hws = int(img_size / 2.)
    if (ix2 < hws+brd
        or ix2>img2.shape[1]-hws-brd-1
        or iy2 < hws+brd
        or iy2>img2.shape[0]-hws-brd-1):
        return np.nan, np.nan, np.nan, np.nan
    image = img2[iy2-hws-brd:iy2+hws+brd+1, ix2-hws-brd:ix2+hws+brd+1]
    if np.any(np.array(image.shape) < (img_size+brd+brd)):
        return np.nan, np.nan, np.nan, np.nan
    r,a,dx,dy,_,_ = rotate_and_match(img1, x1, y1, img_size, image, alpha0, angles)

    x2 = ix2 + dx - 1
    y2 = iy2 + dy - 1

    return x2, y2, r, a

