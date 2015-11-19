# -*- coding: utf-8 -*-
"""
Created on 17 Nov 2015
@author: Stefan Muckenhuber et al. (NERSC)
'Sea ice drift from Sentinel-1 SAR imagery using open source feature tracking'
"""

from nansat import *
import numpy as np
from gdal import *
import cv2
import datetime

def s1a_icedrift(n_a,n_b,band='HV',nFeatures=100000,patchSize=34,ratio_test=0.75,resize_factor=0.5):    
    '''
    Function to derive sea ice drift vectors from Sentinel-1 SAR image pairs
    Input:
    n_a,n_b ... Nansat objects of 2 consecutive, overlapping Sentinel-1 SAR images
    band ... Polarisation of SAR image ('HH' or 'HV')
    nFeatures ... Amount of maximal retained keypoints (recommended: 100000)
    patchSize ... Size of considered patch for feature description (recommended: 34)
    ratio_test ... Threshold for Lowe ratio test (recommended: 0.75)
    resize_factor ... Resolution reduction during pre-processing (recommended: 0.5)
    Return:
    query_lonlat ... Longitute and Latitude for start of drift vector
    train_lonlat ... Longitute and Latitude for end of drift vector
    '''
    
    # Reduce spatial resolution
    n_a.resize(resize_factor, eResampleAlg=-1)
    n_b.resize(resize_factor, eResampleAlg=-1)
    
    # Read in Sentinel-1 images as numpy arrays
    npy_a=n_a['sigma0_'+str(band)]
    npy_b=n_b['sigma0_'+str(band)]
    
    # Backscatter limits
    sigma0_min = {'HH': 0, 'HV': 0}
    sigma0_max = {'HH': 0.08,   'HV': 0.013}
  
    # Speed limits derived from acpuisiton time
    time_a=datetime.datetime(int(file_a[17:21]),int(file_a[21:23]),int(file_a[23:25]),int(file_a[26:28]),int(file_a[28:30]),int(file_a[30:32]))
    time_b=datetime.datetime(int(file_b[17:21]),int(file_b[21:23]),int(file_b[23:25]),int(file_b[26:28]),int(file_b[28:30]),int(file_b[30:32]))
    time_delta=(time_b-time_a).days*24+((time_b-time_a).seconds)/3600 # hours
    speed_low=-1 # meters
    speed_high=10000+1000*time_delta # meters
    
    # Initiate ORB detector
    detector = cv2.ORB()
    detector.setInt('edgeThreshold',patchSize)
    detector.setInt('nFeatures',nFeatures)
    detector.setInt('nLevels',7)
    detector.setInt('patchSize',patchSize)
    print 'ORB detector initiated'
    
    # Compute keypoints
    p12 = [npy_a, npy_b]
    img12 = []
    imgkp12 = []
    kp12 = []
    descr12 = []
    for p in p12:
        # redistribute into range [0,255]
        img = 255 * (p - sigma0_min[band]) / (sigma0_max[band] - sigma0_min[band])
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype('uint8')
        img12.append(img)
        # find keypoints and compute descriptors with ORB
        kp, descr = detector.detectAndCompute(img,None)
        kp12.append(kp)
        descr12.append(descr)
    print 'Keypoints computed with ORB'
    
    # Match keypoints using BFMatcher with cv2.NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descr12[0],descr12[1], k=2)
    print 'Keypoints matched'
    
    # Apply ratio test from Lowe
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    print 'Ratio test applied with '+str(ratio_test)
    
    # Coordinates for start, end point of vectors
    queryX = np.array([kp12[0][m.queryIdx].pt[0] for m in good])
    queryY = np.array([kp12[0][m.queryIdx].pt[1] for m in good])
    trainX = np.array([kp12[1][m.trainIdx].pt[0] for m in good])
    trainY = np.array([kp12[1][m.trainIdx].pt[1] for m in good])
    
    # get approximate center of the scene
    cornersLon, cornersLat = n_a.get_corners()
    lon_0, lat_0 = cornersLon.mean(), cornersLat.mean()
    # reproject Ground Control Points (GCPS) to stereographic projection
    n_a.reproject_GCPs('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon_0, lat_0))
    # get approximate center of the scene
    cornersLon, cornersLat = n_b.get_corners()
    lon_0, lat_0 = cornersLon.mean(), cornersLat.mean()
    # reproject Ground Control Points (GCPS) to stereographic projection
    n_b.reproject_GCPs('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon_0, lat_0))
    print 'Reprojected on stereographic projection'

    # Apply spine interpolation
    n_a.vrt.tps = True
    n_b.vrt.tps = True
    print 'Use spline interpolation'
    
    # Transform oordinates into longitude, latitude
    query_lonlat=n_a.transform_points(queryX, queryY)
    train_lonlat=n_b.transform_points(trainX, trainY)
    
    # Calculate drift speed
    spd = []
    for i in range(query_lonlat[0].size):
        # Equirectangular approximation
        dlong = (train_lonlat[0][i] - query_lonlat[0][i])*np.pi/180;
        dlat  = (train_lonlat[1][i] - query_lonlat[1][i])*np.pi/180;
        slat  = (train_lonlat[1][i] + query_lonlat[1][i])*np.pi/180;
        p1 = (dlong)*np.cos(0.5*slat)
        p2 = (dlat)  
        spd.append(6371000 * np.sqrt( p1*p1 + p2*p2))
    speed=np.array(spd)
    
    # Apply speed filter
    gpi = (speed > speed_low) * (speed < speed_high)
    query_lonlat=n_a.transform_points(queryX[gpi], queryY[gpi])
    train_lonlat=n_b.transform_points(trainX[gpi], trainY[gpi])
    print 'Speed filter applied with '+str(speed_low/1000.)+' - '+str(speed_high/1000.)+' km'
    
    n_a.undo()
    n_b.undo()
    
    return query_lonlat, train_lonlat









