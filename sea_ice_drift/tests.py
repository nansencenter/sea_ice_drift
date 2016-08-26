from __future__ import absolute_import

import os
import sys
import glob
import unittest

import numpy as np
import matplotlib.pyplot as plt

from nansat import Nansat, NSR
from sea_ice_drift import (get_uint8_image, find_key_points,
                           get_match_coords, lstsq_filter,
                           reproject_gcp_to_stere)

class IceDriftTest(unittest.TestCase):
    def setUp(self):
        ''' Load test data '''
        testDir = os.getenv('ICE_DRIFT_TEST_DATA_DIR')
        if testDir is None:
            sys.exit('ICE_DRIFT_TEST_DATA_DIR is not defined')
        testFiles = sorted(glob.glob(os.path.join(testDir, 'S1A_*tif')))
        if len(testFiles) < 2:
            sys.exit('Not enough test files in %s' % testDir)
        self.n1 = Nansat(testFiles[0])
        self.n2 = Nansat(testFiles[1])
        self.img1 = self.n1['sigma0_HV']
        self.img2 = self.n2['sigma0_HV']
        self.imgMin = 0.001
        self.imgMax = 0.013

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
        plt.imsave('0_imgUint8.png', imageUint8, vmin=0, vmax=255)

        self.assertEqual(imageUint8.dtype, np.uint8)
        self.assertEqual(imageUint8.min(), 0)
        self.assertEqual(imageUint8.max(), 255)

    def test_find_key_points(self):
        ''' Shall find key points using default values '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1)

        self.assertTrue(len(keyPoints1) > 1000)

    def test_get_match_coords(self):
        ''' Shall find matching coordinates '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=10000)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=10000)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

    def test_lstsq_filter(self):
        ''' Shall filter out not matching points '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=10000)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=10000)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

        goodPixels = lstsq_filter(x1, y1, x2, y2)
        self.assertTrue(len(goodPixels) > len(goodPixels[goodPixels]))

if __name__ == '__main__':
    unittest.main()
