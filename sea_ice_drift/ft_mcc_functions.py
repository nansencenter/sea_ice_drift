# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:15:16 2016

@author: stemuc
"""

import os
import numpy as np
import cv2
from cv2 import matchTemplate
from nansat import *
from scipy.ndimage.interpolation import rotate
#from geostatsmodels import utilities, variograms, model, kriging, geoplot
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate
from scipy import ndimage as nd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def reproject_gcp_to_stere(n):
    ''' Change projection of GCPs to stereographic add TPS option '''
    lon, lat = n.get_border()
    # reproject Ground Control Points (GCPS) to stereographic projection
    n.reproject_GCPs('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon.mean(), lat.mean()))
    n.vrt.tps = True

    return n

# convert to gray levels image
def grey_levels(image, min_db, max_db):
    gl = 255
    image[image==0]=0.00000001
    img_gl = gl * (np.log10(image) - min_db) / (max_db - min_db)
    img_gl[img_gl < 0] = 0
    img_gl[img_gl > gl] = gl
    img_gl = np.floor(img_gl)
    img_gl = 255 * img_gl / gl
    return img_gl.astype('uint8')

def find_key_points(image, detector=cv2.ORB,
                    edgeThreshold=34,
                    nFeatures=100000,
                    nLevels=7,
                    patchSize=34,
                    **kwargs):
    ''' Initiate detector and find key points on an image '''

    detector = detector()
    detector.setInt('edgeThreshold', edgeThreshold)
    detector.setInt('nFeatures', nFeatures)
    detector.setInt('nLevels', nLevels)
    detector.setInt('patchSize', patchSize)
    print 'ORB detector initiated'

    keyPoints, descriptors = detector.detectAndCompute(image, None)
    print 'Key point found'
    return keyPoints, descriptors


def get_match_coords(keyPoints1, descriptors1,
                                    keyPoints2, descriptors2,
                                    matcher=cv2.BFMatcher,
                                    norm=cv2.NORM_HAMMING,
                                    ratio_test=0.75,
                                    **kwargs):
    ''' Filter matching keypoints and convert to X,Y coordinates '''
    # Match keypoints using BFMatcher with cv2.NORM_HAMMING
    bf = matcher(norm)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    print 'Keypoints matched'

    # Apply ratio test from Lowe
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    print 'Ratio test %f found %d keypoints' % (ratio_test, len(good))

    # Coordinates for start, end point of vectors
    x1 = np.array([keyPoints1[m.queryIdx].pt[0] for m in good])
    y1 = np.array([keyPoints1[m.queryIdx].pt[1] for m in good])
    x2 = np.array([keyPoints2[m.trainIdx].pt[0] for m in good])
    y2 = np.array([keyPoints2[m.trainIdx].pt[1] for m in good])

    return x1, y1, x2, y2

def get_displacement_km(n1, x1, y1, n2, x2, y2):
    ''' Find displacement in kilometers using Domain'''
    lon1, lat1 = n1.transform_points(x1, y1)
    lon2, lat2 = n2.transform_points(x2, y2)

    d = Domain('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon1.mean(), lat1.mean()),
                '-te -100000 -100000 100000 100000 -tr 1000 1000')

    x1d, y1d = d.transform_points(lon1, lat1, 1)
    x2d, y2d = d.transform_points(lon2, lat2, 1)

    return x2d - x1d, y1d - y2d

def get_displacement_pix(n1, x1, y1, n2, x2, y2):
    ''' Find displacement in pixels of the first image'''
    lon2, lat2 = n2.transform_points(x2, y2)
    x2n1, y2n1 = n1.transform_points(lon2, lat2, 1)

    return x2n1 - x1, y2n1 - y1

def remove_too_large(u, v, lon1, lat1, lon2, lat2, maxSpeed):
    ''' filter too high u, v '''
    gpi = np.hypot(u, v) < maxSpeed
    u = u[gpi]
    v = v[gpi]
    lon1 = lon1[gpi]
    lat1 = lat1[gpi]
    lon2 = lon2[gpi]
    lat2 = lat2[gpi]

    return u, v, lon1, lat1, lon2, lat2

def lstsq_filter(x1, y1, x2, y2, psi=600, **kwargs):
    ''' Remove vectors that don't fit the model x1 = f(x2, y2)^n

    Fit the model x1 = f(x2, y2)^n using least squares method
    Simulate x1 using the model
    Compare actual and simulated x1 and remove points where error is too high
    Parameters:
        x1, y1, x2, y2 : coordinates of start and end of displacement [pixels]
        psi : threshold error between actual and simulated x1 [pixels]
    '''
    # stack together target coordinates
    A = np.vstack([np.ones(len(x2)), x2, y2, x2**2, y2**2, x2*y2, x2**3, y2**3]).T

    # find B in x1 = B * [x2, y2]
    Bx = np.linalg.lstsq(A, x1)[0]
    By = np.linalg.lstsq(A, y1)[0]

    # calculate simulated x1sim = B * [x2, y2]
    x1sim = np.dot(A, Bx)
    y1sim = np.dot(A, By)

    # find error between actual and simulated x1
    xErr = (x1 - x1sim) ** 2
    yErr = (y1 - y1sim) ** 2

    # find pixels with error below psi
    goodPixels = (xErr < psi ** 2) * (yErr < psi ** 2)

    return goodPixels

def get_n_img(file1, file2, band='HV', resize_factor=0.5, maxSpeed=0.5):
    ''' Load Sentinel-1 images as Nansat objects,
    calcualte image matrixes with decibel scaling and
    rotation between the image pair
    '''
    
    # Load Sentinel-1 images as Nansat objects
    n1 = Nansat(file1)
    n2 = Nansat(file2)
    # increase accuracy
    n1 = reproject_gcp_to_stere(n1)
    n2 = reproject_gcp_to_stere(n2)
    # increase speed
    n1.resize(resize_factor, eResampleAlg=-1)
    n2.resize(resize_factor, eResampleAlg=-1)
    
    # HV
    if band=='HV':
        bandName='sigma0_HV'
        min_db = -3.25; max_db = np.log10(0.013)
    
    # HH
    if band=='HH':
        bandName='sigma0_HH'
        min_db = -2.5; max_db = np.log10(0.08)
    
    # get matrices with data
    img1 = n1[bandName]
    img2 = n2[bandName]
    
    # convert to gray levels (DB)
    img1 = grey_levels(img1, min_db, max_db)
    img2 = grey_levels(img2, min_db, max_db)
    
    # initial rotation alpha
    alpha = get_rotation(n1,n2)
    
    return n1, n2, img1, img2, alpha
    
    
def get_rotation(n1,n2):
    '''Calulate rotation between two Sentinel-1 images'''

    corners_n2_lons, corners_n2_lats = n2.get_corners()
    corner0_n2_x1, corner0_n2_y1 = n1.transform_points([corners_n2_lons[0]], [corners_n2_lats[0]], 1)
    corner1_n2_x1, corner1_n2_y1 = n1.transform_points([corners_n2_lons[1]], [corners_n2_lats[1]], 1)
    b = corner1_n2_x1 - corner0_n2_x1
    a = corner1_n2_y1 - corner0_n2_y1
    alpha = np.arctan2(b,a)
    
    return alpha



def get_icedrift_ft(n1, n2, img1, img2, resize_factor=0.5, maxSpeed=0.5):
    ''' Calculate ice drift from Sentinel-1 image pair using feature tracking
    [Open-source feature-tracking algorithm for sea ice drift retrieval from Sentinel-1 SAR imagery
    Stefan Muckenhuber, Anton Andreevich Korosov, and Stein Sandven
    The Cryosphere, 10, 913-925, doi:10.5194/tc-10-913-2016, 2016]
    '''
    
    # find key points
    kp1, descr1 = find_key_points(img1)
    kp2, descr2 = find_key_points(img2)
    
    # find coordinates of matching key points
    x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2)
    
    # convert x,y to lon, lat
    lon1, lat1 = n1.transform_points(x1, y1)
    lon2, lat2 = n2.transform_points(x2, y2)
    
    # find displacement in kilometers
    u, v = get_displacement_km(n1, x1, y1, n2, x2, y2)
    
    # convert to speed in m/s
    t1 = n1.get_time()[0]
    t2 = n2.get_time()[0]
    dt = t2 - t1
    u = u * 1000 / dt.total_seconds()
    v = v * 1000 / dt.total_seconds()
    
    # filter too high u, v
    u_input, v_input, x1, y1, x2, y2 = remove_too_large(u, v, x1, y1, x2, y2, maxSpeed)
    
    print 'Feature Tracking successfull: '+str(len(x1))+' vectors'
    return x1, y1, x2, y2


def get_GridFSim(x1, y1, x2, y2, img1):
    ''' Calculate estimated ice drift on first image based on feature tracking vectors'''
    
    # # initial drift inter-/extrapolation
    # linear triangulation
    x1Grid, y1Grid = np.meshgrid(range(img1.shape[1]), range(img1.shape[0]))
    x2GridFSim = griddata(np.array([y1, x1]).T, x2, np.array([y1Grid, x1Grid]).T, method='linear').T
    y2GridFSim = griddata(np.array([y1, x1]).T, y2, np.array([y1Grid, x1Grid]).T, method='linear').T
    # linear fit for entire grid
    A = np.vstack([np.ones(len(x1)), x1, y1 ]).T
    # find B in x2 = B * [x1, y1]
    Bx = np.linalg.lstsq(A, x2)[0]
    By = np.linalg.lstsq(A, y2)[0]
    # calculate simulated x2sim = B * [x1, y1]
    x1GridF = x1Grid.flatten()
    y1GridF = y1Grid.flatten()
    A = np.vstack([np.ones(len(x1GridF)), x1GridF, y1GridF]).T
    x2GridFSim_lf = np.dot(A, Bx).reshape(img1.shape)
    y2GridFSim_lf = np.dot(A, By).reshape(img1.shape)
    # fill NaN with lf
    gpi = np.isnan(x2GridFSim)
    x2GridFSim[gpi] = x2GridFSim_lf[gpi]
    y2GridFSim[gpi] = y2GridFSim_lf[gpi]

    return x2GridFSim, y2GridFSim


def get_border_grid(x1, y1, img1, hws_n2_range):
    ''' Calculate distance to nearest feature tracking vector'''
    
    # distance to nearest FT vector
    seed = np.zeros(img1.shape, dtype=bool)
    seed[y1.astype(np.uint16), x1.astype(np.uint16)] = True
    dist = nd.distance_transform_edt(~seed, return_distances=True, return_indices=False)
    
    # border_grid
    minborder = hws_n2_range[0]
    maxborder = hws_n2_range[1]
    border_grid = dist.copy()
    border_grid[border_grid < minborder] = minborder
    border_grid[border_grid > maxborder] = maxborder
    
    return border_grid



def get_icedrift_mcc_pois(n1, n2, img1, img2, alpha, x1raw, y1raw, x2raw, y2raw,
                          pois, hws_n1 = 35, hws_n2_range = [20, 125], minMCC = 0.35,
                          angles = np.arange(-10, 10.1, 2)):
    ''' Calculate ice drift on Sentinel-1 image pair at points of interest
    using pattern matching and based on initial drift estimate from feature tracking
    '''

    # pois in n1 coordinates
    x1_pois, y1_pois = n1.transform_points(pois[0], pois[1], -1)

    # fitler outlier
    gpi = lstsq_filter(x1raw,y1raw,x2raw,y2raw, 100)
    x1 = x1raw[gpi]; y1 = y1raw[gpi]; x2 = x2raw[gpi]; y2 = y2raw[gpi]
    
    # initial drift inter-/extrapolation
    x2GridFSim, y2GridFSim = get_GridFSim(x1, y1, x2, y2, img1)
    
    # get border_grid
    border_grid = get_border_grid(x1, y1, img1, hws_n2_range)
    
    # initialize 
    r2mcc = np.zeros(len(x1_pois)) + np.nan
    c2mcc = np.zeros(len(x1_pois)) + np.nan
    dr2mcc = np.zeros(len(x1_pois)) + np.nan
    dc2mcc = np.zeros(len(x1_pois)) + np.nan
    mccr = np.zeros(len(x1_pois)) + np.nan
    rot = np.zeros(len(x1_pois)) + np.nan
    
    for i in range(len(x1_pois)):
        r1 = y1_pois[i]
        c1 = x1_pois[i]
        # FIRST GUESS
        r2 = y2GridFSim[r1, c1]
        c2 = x2GridFSim[r1, c1]
        
        # BORDER
        border = border_grid[r1, c1]
        
        # Check if inside image boundaries
        if (np.isnan(r2) or
            np.isnan(c2) or
            r2-hws_n1-border < 10 or
            c2-hws_n1-border < 10 or
            r2+hws_n1+border+10 > img2.shape[0] or
            c2+hws_n1+border+10 > img2.shape[1] or
            np.isnan(r1) or
            np.isnan(c1) or
            r1-hws_n1 < 10 or
            c1-hws_n1 < 10 or
            r1+hws_n1+10 > img1.shape[0] or
            c1+hws_n1+10 > img1.shape[1]):
            continue
    
        # large part of image 2
        template_n2 = img2[r2-hws_n1-border:r2+hws_n1+border, c2-hws_n1-border:c2+hws_n1+border]
        
        # MCC with rotation
        # rotate and match
        best_r = 0
        best_a = 0
        best_ij = (0,0)
        for angle in angles:            
            angle_rad = np.radians(np.degrees(alpha)+angle)
            
            # large template to account for rotation
            hws_n1_large = np.ceil(hws_n1 * np.abs(np.cos(angle_rad)) + hws_n1 * np.abs(np.sin(angle_rad)))
            template_n1_large = img1[r1-hws_n1_large:r1+hws_n1_large, c1-hws_n1_large:c1+hws_n1_large]
            
            # half size of template after rotation
            hws_n1_large2 = np.ceil(hws_n1_large * np.abs(np.cos(angle_rad)) + hws_n1_large * np.abs(np.sin(angle_rad)))
            # margin of rotated template and central part
            rotBorder1 = hws_n1_large2 - hws_n1
            rotBorder2 = rotBorder1 + hws_n1 + hws_n1
            
            # rotate according to angle_rad
            template_n1_rotate = rotate(template_n1_large, -np.degrees(angle_rad), order=1)

            # final n1
            template_n1 = template_n1_rotate[rotBorder1:rotBorder2, rotBorder1:rotBorder2].astype('uint8')
            
            ### MATCHING ONE SINGLE ROTATION
            result = matchTemplate(template_n2, template_n1, cv2.TM_CCOEFF_NORMED)
    
            ij = np.unravel_index(np.argmax(result), result.shape)
            if result.max() > best_r:
                best_r = result.max()
                best_a = angle
                best_ij = ij
        
        dr2mcc[i] = best_ij[0] - border
        dc2mcc[i] = best_ij[1] - border
    
        r2mcc[i] = r2 + best_ij[0] - border
        c2mcc[i] = c2 + best_ij[1] - border
    
        mccr[i] = best_r
        rot[i] = best_a
            
    dr2mcc[mccr < minMCC] = np.nan
    dc2mcc[mccr < minMCC] = np.nan
    r2mcc[mccr < minMCC] = np.nan
    c2mcc[mccr < minMCC] = np.nan
    rot[mccr < minMCC] = np.nan
    
    # FT (first guess)
    lon1_fg, lat1_fg = pois[0], pois[1]
    x2_fg = np.zeros(len(x1_pois)) + np.nan
    y2_fg = np.zeros(len(x1_pois)) + np.nan
    for i in range(len(x1_pois)):
        x2_fg[i] = x2GridFSim[y1_pois[i], x1_pois[i]]
        y2_fg[i] = y2GridFSim[y1_pois[i], x1_pois[i]] 
    lon2_fg, lat2_fg = n2.transform_points(x2_fg, y2_fg)
    
    # FT + MCC
    gpi = np.isfinite(c2mcc)
    lon1_mcc, lat1_mcc = np.array(pois[0])[gpi], np.array(pois[1])[gpi]
    lon2_mcc, lat2_mcc = n2.transform_points(c2mcc[gpi], r2mcc[gpi])
    
    return lon1_fg, lat1_fg, lon2_fg, lat2_fg, lon1_mcc, lat1_mcc, lon2_mcc, lat2_mcc, mccr[gpi]
    


def get_icedrift_mcc_grid(n1, n2, img1, img2, alpha, x1raw, y1raw, x2raw, y2raw,
                          hws_n1 = 35, hws_n2_range = [20, 125], minMCC = 0.35,
                          grid_step = 100, angles = np.arange(-10, 10.1, 2)):
    ''' Calculate ice drift on Sentinel-1 image pair at grid on first image
    using pattern matching and based on initial drift estimate from feature tracking
    '''
    # vector grid
    tmpborder = int(np.ceil(hws_n1*np.cos(np.pi/4) + hws_n1*np.sin(np.pi/4))+1)
    rows1 = range(tmpborder, img1.shape[0]-tmpborder, grid_step)
    cols1 = range(tmpborder, img1.shape[1]-tmpborder, grid_step)
    pois_xy = []
    for r1 in rows1:
        for c1 in cols1:
            pois_xy.append([c1,r1])
    pois_xy = np.array(pois_xy)
    pois = n1.transform_points(pois_xy[:,0], pois_xy[:,1])
    lon1_fg, lat1_fg, lon2_fg, lat2_fg, lon1_mcc, lat1_mcc, lon2_mcc, lat2_mcc, mccr = get_icedrift_mcc_pois(n1, n2, img1, img2, alpha, x1raw, y1raw, x2raw, y2raw,
                                                                                                             pois, hws_n1 = 35, hws_n2_range = [20, 125], minMCC = 0.35,
                                                                                                             angles = np.arange(-10, 10.1, 2))
    
    return lon1_fg, lat1_fg, lon2_fg, lat2_fg, lon1_mcc, lat1_mcc, lon2_mcc, lat2_mcc, mccr
    


def get_u_v(query_lon,query_lat,train_lon,train_lat):
    ''' Calculate u and v displacemnt based on start and end position in Lon/Lat'''
    # Equirectangular approximation
    dlong = (train_lon - query_lon)*np.pi/180;
    dlat  = (train_lat - query_lat)*np.pi/180;
    slat  = (train_lat + query_lat)*np.pi/180;
    p1 = (dlong)*np.cos(0.5*slat)
    p2 = (dlat)  
    u = 6371000 * p1
    v = 6371000 * p2
    return u, v


def get_validation_result(lon1_val, lat1_val, lon2_val, lat2_val, lon1_mcc, lat1_mcc, lon2_mcc, lat2_mcc, mccr):
    
    # N ... number of good maches MCC
    N = len(mccr)
    # average MCC value of good matches
    MCC_average = np.mean(mccr)
    
    # MCC vectors
    query_lonlat = [lon1_mcc, lat1_mcc]
    train_lonlat = [lon2_mcc, lat2_mcc]
    
    
    # Validation vectors
    lon1 = lon1_val
    lat1 = lat1_val
    lon2 = lon2_val
    lat2 = lat2_val
    
    min_ind=[]
    min_distance=[]
    for j in range(len(lon1)):
        # dist(ance) in meters to all points
        spd = []
        for i in range(query_lonlat[0].size):
            # Equirectangular approximation
            dlong = (float(lon1[j]) - query_lonlat[0][i])*np.pi/180;
            dlat  = (float(lat1[j]) - query_lonlat[1][i])*np.pi/180;
            slat  = (float(lat1[j]) + query_lonlat[1][i])*np.pi/180;
            p1 = (dlong)*np.cos(0.5*slat)
            p2 = (dlat)  
            spd.append(6371000 * np.sqrt( p1*p1 + p2*p2))
    #    dist=np.array(spd)
        min_ind.append(spd.index(min(spd)))
        min_distance.append(np.nanmin(spd))
    #print 'Mean distance = '+str(np.nanmean(min_distance))+'+/-'+str(np.nanstd(min_distance))
    
    query_lon = query_lonlat[0][min_ind]
    query_lat = query_lonlat[1][min_ind]
    train_lon = train_lonlat[0][min_ind]
    train_lat = train_lonlat[1][min_ind]
    
    
    # distance vec_1 to vec_2 [m]
    spd = []
    for i in range(query_lon.size):
        # Equirectangular approximation
        dlong = (float(lon1[i]) - query_lon[i])*np.pi/180;
        dlat  = (float(lat1[i]) - query_lat[i])*np.pi/180;
        slat  = (float(lat1[i]) + query_lat[i])*np.pi/180;
        p1 = (dlong)*np.cos(0.5*slat)
        p2 = (dlat)  
        spd.append(6371000 * np.sqrt( p1*p1 + p2*p2))
    
    spd=np.array(spd)
    # limit maximal distance [m]
    b=spd<100
    r = np.array(range(len(b)))
    index_5km = r[b]
    N_0_distance = len(index_5km)
    #print 'Number of vectors with 0 distance = '+str(len(index_5km))
    
    # speed in meters - MCC
    u = []
    v = []
    for i in index_5km:#range(query_lon.size):
        # Equirectangular approximation
        dlong = (train_lon[i] - query_lon[i])*np.pi/180;
        dlat  = (train_lat[i] - query_lat[i])*np.pi/180;
        slat  = (train_lat[i] + query_lat[i])*np.pi/180;
        p1 = (dlong)*np.cos(0.5*slat)
        p2 = (dlat)  
        u.append(6371000 * p1)
        v.append(6371000 * p2)
    
    u=np.array(u)
    v=np.array(v)
    
    # speed in meters - validation
    u_val = []
    v_val = []
    for i in index_5km:#range(lon1.size):
        # Equirectangular approximation
        dlong = (lon2[i] - lon1[i])*np.pi/180;
        dlat  = (lat2[i] - lat1[i])*np.pi/180;
        slat  = (lat2[i] + lat1[i])*np.pi/180;
        p1 = (dlong)*np.cos(0.5*slat)
        p2 = (dlat)  
        u_val.append(6371000 * p1)
        v_val.append(6371000 * p2)
    
    u_val=np.array(u_val)
    v_val=np.array(v_val)
    
    # linear fit
    x=[]
    y=[]
    for i in range(len(index_5km)):
        x.append(np.sqrt(u[i]**2+v[i]**2))
        y.append(np.sqrt(u_val[i]**2+v_val[i]**2))
        
    x=np.array(x)
    y=np.array(y)
    inclination, offset = np.polyfit(x,y,1)
    #print 'Inclination = '+str(inclination)
    #print 'Offset = '+str(offset)
    
    # RMSD
    rd_i=[]
    for i in range(u.size):
        rd_i.append((u[i]-u_val[i])**2+(v[i]-v_val[i])**2)
    
    rmsd=np.sqrt(np.nansum(rd_i)/len(rd_i))
    #print 'RMSD = '+str(rmsd)
    
    print 'N, N_0_distance, MCC_average, rmsd, inclination, offset'
    print str(N)+', '+str(N_0_distance)+', '+str(MCC_average)+', '+str(rmsd)+', '+str(inclination)+', '+str(offset)

    return N, N_0_distance, MCC_average, rmsd, inclination, offset


def plot_FG_border(x1raw, y1raw, x2raw, y2raw, img1, hws_n1 = 35, hws_n2_range = [20, 125]):

    # plot x2GridFSim, y2GridFSim and border-grid
    hws_n1 = 35; hws_n2_range = [20, 125]
    # fitler outlier
    gpi = lstsq_filter(x1raw,y1raw,x2raw,y2raw, 100)
    print len(gpi), len(gpi[gpi])
    x1 = x1raw[gpi]
    y1 = y1raw[gpi]
    x2 = x2raw[gpi]
    y2 = y2raw[gpi]
    # initial drift inter-/extrapolation
    x2GridFSim, y2GridFSim = get_GridFSim(x1, y1, x2, y2, img1)
    # get border_grid
    border_grid = get_border_grid(x1, y1, img1, hws_n2_range)
    
    plt.figure(num=1, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1,3,1)
    ax = plt.gca()
    #im = ax.imshow(np.arange(100).reshape((10,10)))
    im = plt.imshow(x2GridFSim)#;plt.colorbar(orientation='horizontal')
    plt.axis('off')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1,3,2)
    ax = plt.gca()
    #im = ax.imshow(np.arange(100).reshape((10,10)))
    im = plt.imshow(y2GridFSim)#;plt.colorbar(orientation='horizontal')
    plt.axis('off')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1,3,3)
    ax = plt.gca()
    #im = ax.imshow(np.arange(100).reshape((10,10)))
    im = plt.imshow(border_grid)#;plt.colorbar(orientation='horizontal')
    plt.axis('off')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    return plt