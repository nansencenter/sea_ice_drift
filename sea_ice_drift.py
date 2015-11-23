import numpy as np
import cv2

from nansat import Domain


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
                    patchSize=34):
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
                                    ratio_test=0.75):
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

def reproject_gcp_to_stere(n):
    ''' Change projection of GCPs to stereographic add TPS option '''
    lon, lat = n.get_border()
    # reproject Ground Control Points (GCPS) to stereographic projection
    n.reproject_GCPs('+proj=stere +lon_0=%f +lat_0=%f +no_defs' % (lon.mean(), lat.mean()))
    n.vrt.tps = True

    return n

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












