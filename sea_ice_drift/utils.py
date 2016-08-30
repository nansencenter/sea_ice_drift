import time
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
    uint8Image[~np.isfinite(uint8Image)] = 0

    return uint8Image.astype('uint8')

def find_key_points(image, detector='ORB',
                    edgeThreshold=34,
                    nFeatures=100000,
                    nLevels=7,
                    patchSize=34,
                    **kwargs):
    ''' Initiate detector and find key points on an image '''
    if detector=='ORB' and cv2.__version__.startswith('3.'):
        detector = cv2.ORB_create()
        detector.setEdgeThreshold(edgeThreshold)
        detector.setMaxFeatures(nFeatures)
        detector.setNLevels(nLevels)
        detector.setPatchSize(patchSize)
    elif detector=='ORB':
        detector = cv2.ORB()
        detector.setInt('edgeThreshold', edgeThreshold)
        detector.setInt('nFeatures', nFeatures)
        detector.setInt('nLevels', nLevels)
        detector.setInt('patchSize', patchSize)
    print 'ORB detector initiated'

    keyPoints, descriptors = detector.detectAndCompute(image, None)
    print 'Key points found: %d' % len(keyPoints)
    return keyPoints, descriptors


def get_match_coords(keyPoints1, descriptors1,
                                    keyPoints2, descriptors2,
                                    matcher=cv2.BFMatcher,
                                    norm=cv2.NORM_HAMMING,
                                    ratio_test=0.75,
                                    **kwargs):
    ''' Filter matching keypoints and convert to X,Y coordinates '''
    t0 = time.time()
    # Match keypoints using BFMatcher with cv2.NORM_HAMMING
    bf = matcher(norm)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    t1 = time.time()
    print 'Keypoints matched', t1 - t0

    # Apply ratio test from Lowe
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    t2 = time.time()
    print 'Ratio test %f found %d keypoints in %f' % (ratio_test, len(good), t2-t1)

    # Coordinates for start, end point of vectors
    x1 = np.array([keyPoints1[m.queryIdx].pt[0] for m in good])
    y1 = np.array([keyPoints1[m.queryIdx].pt[1] for m in good])
    x2 = np.array([keyPoints2[m.trainIdx].pt[0] for m in good])
    y2 = np.array([keyPoints2[m.trainIdx].pt[1] for m in good])

    return x1, y1, x2, y2

def get_displacement_km(n1, x1, y1, n2, x2, y2, ll2km='domain'):
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

def domain_filter(n, keyPoints, descr, domain, domainMargin=0):
    ''' Finds keypoints from Nansat objects <n> which are within <domain>'''
    cols = [kp.pt[0] for kp in keyPoints]
    rows = [kp.pt[1] for kp in keyPoints]
    lon, lat = n.transform_points(cols, rows, 0)
    colsD, rowsD = domain.transform_points(lon, lat, 1)
    gpi = ((colsD >= 0 + domainMargin) *
           (rowsD >= 0 + domainMargin) *
           (colsD <= domain.shape()[1] - domainMargin) *
           (rowsD <= domain.shape()[0] - domainMargin))

    print 'Domain filter: %d -> %d' % (len(keyPoints), len(gpi[gpi]))
    return list(np.array(keyPoints)[gpi]), descr[gpi]

def max_drift_filter(n1, x1, y1, n2, x2, y2, maxDrift=20):
    ''' Filter out too high drift (km) '''
    u, v = get_displacement_km(n1, x1, y1, n2, x2, y2)
    gpi = np.hypot(u,v) <= maxDrift

    print 'MaxDrift filter: %d -> %d' % (len(x1), len(gpi[gpi]))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]

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
    gpi = (xErr < psi ** 2) * (yErr < psi ** 2)

    print 'LSTSQ filter: %d -> %d' % (len(x1), len(gpi[gpi]))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]

def get_denoised_object(filename, bandName, factor):
    from sentinel1denoised.S1_EW_GRD_NoiseCorrection import Sentinel1Image
    s = Sentinel1Image(filename)
    s.add_denoised_band('sigma0_HV')
    s.resize(factor, eResampleAlg=-1)
    img = s[bandName + '_denoised']

    n = Nansat(domain=s)
    n.add_band(img, parameters=s.get_metadata(bandID=bandName))
    n.set_metadata(s.get_metadata())

    return n

class SeaIceDrift(object):
    def __init__(self, filename1, filename2):
        self.filename1 = filename1
        self.filename2 = filename2

    def feature_tracking(self, bandName='sigma0_HV',
                          factor=0.5, vmin=0, vmax=0.013,
                          domainMargin=10, maxDrift=20,
                          denoise=False, dB=False, **kwargs):
        ''' Find starting and ending point of drift using feature tracking '''
        if denoise:
            # open, denoise and reduce size
            n1 = get_denoised_object(self.filename1, bandName, factor)
            n2 = get_denoised_object(self.filename2, bandName, factor)
        else:
            # open and reduce size
            n1 = Nansat(self.filename1)
            n2 = Nansat(self.filename2)
            n1.resize(factor, eResampleAlg=-1)
            n2.resize(factor, eResampleAlg=-1)

        # increase accuracy of coordinate transfomation
        n1 = reproject_gcp_to_stere(n1)
        n2 = reproject_gcp_to_stere(n2)

        # get matrices with data
        img1 = n1[bandName]
        img2 = n2[bandName]

        if not denoise and dB:
            img1 = 10 * np.log10(img1)
            img2 = 10 * np.log10(img2)

        # convert to 0 - 255
        img1 = get_uint8_image(img1, vmin, vmax)
        img2 = get_uint8_image(img2, vmin, vmax)

        # find many key points
        kp1, descr1 = find_key_points(img1, **kwargs)
        kp2, descr2 = find_key_points(img2, **kwargs)

        # filter keypoints by Domain
        kp1, descr1 = domain_filter(n1, kp1, descr1, n2, domainMargin)
        kp2, descr2 = domain_filter(n2, kp2, descr2, n1, domainMargin)

        # find coordinates of matching key points
        x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2, **kwargs)

        # filter out pair with too high drift
        x1, y1, x2, y2 = max_drift_filter(n1, x1, y1, n2, x2, y2, maxDrift)

        # filter out inconsistent pairs
        x1, y1, x2, y2 = lstsq_filter(x1, y1, x2, y2, **kwargs)

        return n1, img1, x1, y1, n2, img2, x2, y2

    def get_drift_vectors(self, n1, x1, y1, n2, x2, y2, ll2km='domain'):
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
