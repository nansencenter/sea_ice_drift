import numpy as np
import cv2

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

    return uint8Image.astype('uint8')

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


class SeaIceDrift(Nansat):
    def get_drift_vectors(n1, n2, bandName='sigma0_HV',
                          factor=0.5, vmin=0, vmax=0.013,
                          maxSpeed=0.5, **kwargs):
        ''' Estimate drift of features between two images '''
        # increase speed
        n1 = reproject_gcp_to_stere(n1)
        n2 = reproject_gcp_to_stere(n2)

        # increase speed
        n1.resize(factor, eResampleAlg=-1)
        n2.resize(factor, eResampleAlg=-1)

        # get matrices with data
        img1 = n1[bandName]
        img2 = n2[bandName]

        # convert to 0 - 255
        img1 = get_uint8_image(img1, vmin, vmax)
        img2 = get_uint8_image(img2, vmin, vmax)

        # find many key points
        kp1, descr1 = find_key_points(img1, **kwargs)
        kp2, descr2 = find_key_points(img2, **kwargs)

        # find coordinates of matching key points
        x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2, **kwargs)

        # filter out inconsistent pairs
        goodPixels = lstsq_filter(x1, y1, x2, y2, **kwargs)
        x1 = x1[goodPixels]
        y1 = y1[goodPixels]
        x2 = x2[goodPixels]
        y2 = y2[goodPixels]

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
        u, v, lon1, lat1, lon2, lat2 = remove_too_large(u, v, lon1, lat1,
                                                              lon2, lat2,
                                                              maxSpeed)

        return u, v, lon1, lat1, lon2, lat2
