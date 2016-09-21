import time

import numpy as np

from scipy import ndimage as nd
from scipy.interpolate import griddata

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
                                    verbose=True,
                                    **kwargs):
    ''' Filter matching keypoints and convert to X,Y coordinates '''
    t0 = time.time()
    # Match keypoints using BFMatcher with cv2.NORM_HAMMING
    bf = matcher(norm)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    t1 = time.time()
    if verbose:
        print 'Keypoints matched', t1 - t0

    # Apply ratio test from Lowe
    good = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:
            good.append(m)
    t2 = time.time()
    if verbose:
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

def lstsq_filter(x1, y1, x2, y2, psi=200, order=2, **kwargs):
    ''' Remove vectors that don't fit the model x1 = f(x2, y2)^n

    Fit the model x1 = f(x2, y2)^n using least squares method
    Simulate x1 using the model
    Compare actual and simulated x1 and remove points where error is too high
    Parameters:
        x1, y1, x2, y2 : coordinates of start and end of displacement [pixels]
        psi : threshold error between actual and simulated x1 [pixels]
    '''
    # interpolate using N-order polynomial
    x2sim, y2sim = x2y2_interpolation_poly(x1, y1, x2, y2, x1, y1, order=order)

    # find error between actual and simulated x1
    err = np.hypot(x2 - x2sim, y2 - y2sim)

    # find pixels with error below psi
    gpi = err < psi

    print 'LSTSQ filter: %d -> %d' % (len(x1), len(gpi[gpi]))
    return x1[gpi], y1[gpi], x2[gpi], y2[gpi]

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
    ''' Calcalate angle of rotation between two images'''
    corners_n2_lons, corners_n2_lats = n2.get_corners()
    corner0_n2_x1, corner0_n2_y1 = n1.transform_points([corners_n2_lons[0]], [corners_n2_lats[0]], 1)
    corner1_n2_x1, corner1_n2_y1 = n1.transform_points([corners_n2_lons[1]], [corners_n2_lats[1]], 1)
    b = corner1_n2_x1 - corner0_n2_x1
    a = corner1_n2_y1 - corner0_n2_y1
    alpha = np.degrees(np.arctan2(b, a)[0])
    return alpha

def rotate_and_match(img1, x, y, img_size, image, alpha0=0, angles=[0],
                     mtype=cv2.TM_CCOEFF_NORMED):
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
            n1 = get_denoised_object(self.filename1, bandName, factor, **kwargs)
            n2 = get_denoised_object(self.filename2, bandName, factor, **kwargs)
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
