from __future__ import absolute_import

import os
import sys
import glob
import unittest
import inspect

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from nansat import Nansat, NSR
from sea_ice_drift import (SeaIceDrift,
                           get_uint8_image,
                           find_key_points,
                           get_match_coords,
                           domain_filter,
                           max_drift_filter,
                           lstsq_filter,
                           reproject_gcp_to_stere,
                           get_displacement_km,
                           get_displacement_pix,
                           get_n_img)

class SeaIceDriftLibTests(unittest.TestCase):
    def setUp(self):
        ''' Load test data '''
        testDir = os.getenv('ICE_DRIFT_TEST_DATA_DIR')
        if testDir is None:
            sys.exit('ICE_DRIFT_TEST_DATA_DIR is not defined')
        self.testFiles = sorted(glob.glob(os.path.join(testDir, 'S1A_*tif')))
        if len(self.testFiles) < 2:
            sys.exit('Not enough test files in %s' % testDir)
        self.n1 = Nansat(self.testFiles[0])
        self.n2 = Nansat(self.testFiles[1])
        self.img1 = self.n1['sigma0_HV']
        self.img2 = self.n2['sigma0_HV']
        self.imgMin = 0.001
        self.imgMax = 0.013
        self.nFeatures = 10000

    def test_reproject_gcp_to_stere(self):
        ''' Shall change projection of GCPs to stere '''
        n1pro = reproject_gcp_to_stere(self.n1)

        self.assertTrue(n1pro.vrt.tps)
        self.assertTrue(len(n1pro.vrt.dataset.GetGCPs()) > 0)
        self.assertTrue((NSR(n1pro.vrt.dataset.GetGCPProjection())
                            .ExportToProj4().startswith('+proj=stere')))

    def test_get_uint8_image(self):
        ''' Shall scale image values from float (or any) to 0 - 255 [uint8] '''

        imageUint8 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        plt.imsave('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,
                    imageUint8, vmin=0, vmax=255)

        self.assertEqual(imageUint8.dtype, np.uint8)
        self.assertEqual(imageUint8.min(), 0)
        self.assertEqual(imageUint8.max(), 255)

    def test_get_displacement_km(self):
        ''' Shall find matching coordinates and plot quiver in lon/lat'''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        u, v = get_displacement_km(self.n1, x1, y1, self.n2, x2, y2)
        lon1, lat1 = self.n1.transform_points(x1, y1)

        plt.quiver(lon1, lat1, u, v)
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertTrue(len(u) == len(x1))

    def test_get_displacement_pix(self):
        ''' Shall find matching coordinates and plot quiver in pixel/line'''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        u, v = get_displacement_pix(self.n1, x1, y1, self.n2, x2, y2)

        plt.quiver(x1, y1, u, v)
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertTrue(len(u) == len(x1))

    def test_get_n_img(self):
        ''' Shall return Nansat and Matrix '''
        n, img = get_n_img(self.testFiles[0],
                           'sigma0_HV', 0.5, 0.001, 0.013, False, False)
        
        self.assertIsInstance(n, Nansat)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.min(), 0)
        self.assertEqual(img.max(), 255)

class SeaIceDriftFTLibTests(SeaIceDriftLibTests):
    def test_find_key_points(self):
        ''' Shall find key points using default values '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1)

        self.assertTrue(len(keyPoints1) > 1000)

    def test_get_match_coords(self):
        ''' Shall find matching coordinates '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        self.assertTrue(len(keyPoints1) > len(x1))
        self.assertTrue(len(keyPoints2) > len(x2))

    def test_domain_filter(self):
        ''' Shall leave keypoints from second image withn domain of the first '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)

        keyPoints2f, descr2f = domain_filter(self.n2, keyPoints2, descr2,
                                             self.n1, domainMargin=100)

        # plot dots
        cols1 = [kp.pt[0] for kp in keyPoints1]
        rows1 = [kp.pt[1] for kp in keyPoints1]
        lon1, lat1 = self.n1.transform_points(cols1, rows1, 0)
        cols2 = [kp.pt[0] for kp in keyPoints2f]
        rows2 = [kp.pt[1] for kp in keyPoints2f]
        lon2, lat2 = self.n2.transform_points(cols2, rows2, 0)
        plt.plot(lon1, lat1, '.')
        plt.plot(lon2, lat2, '.')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')

        self.assertTrue(len(descr2f) < len(descr2))

    def test_max_drift_filter(self):
        '''Shall keep only slow drift '''
        maxSpeed = 30 # km
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        x1f, y1f, x2f, y2f = max_drift_filter(self.n1, x1, y1,
                                          self.n2, x2, y2)

        self.assertTrue(len(x1f) < len(x1))

    def test_lstsq_filter(self):
        ''' Shall filter out not matching points '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

        x1f, y1f, x2f, y2f = lstsq_filter(x1, y1, x2, y2)
        self.assertTrue(len(x1) > len(x1f))


class SeaIceDriftClassTests(SeaIceDriftLibTests):
    def test_feature_tracking(self):
        ''' Shall use all developed functions '''
        sid = SeaIceDrift(self.testFiles[0], self.testFiles[1])
        n1, img1, x1, y1, n2, img2, x2, y2 = sid.feature_tracking()
        u, v, lon1, lat1, lon2, lat2 = sid.get_drift_vectors(n1, x1, y1,
                                                             n2, x2, y2)

        plt.plot(lon1, lat1, '.')
        plt.plot(lon2, lat2, '.')
        plt.quiver(lon1, lat1, u, v, color='r')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()

