from __future__ import absolute_import

import os
import unittest

import numpy as np
import matplotlib.pyplot as plt

from sea_ice_drift import get_uint8_image, find_key_points
from sea_ice_drift import get_match_coords, lstsq_filter


class IceDriftTest(unittest.TestCase):
    def setUp(self):
        ''' Load test data '''
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.testdata_path = os.path.join(self.path, 'testdata')
        self.img1 = np.load(os.path.join(self.testdata_path,
                                         'S1A_EW_GRDM_1SDH_20150328T074433_20150328T074533_005229_0069A8_801E.npz'))['img']
        self.img2 = np.load(os.path.join(self.testdata_path,
                                         'S1A_EW_GRDM_1SDH_20150329T163452_20150329T163552_005249_006A15_FD89.npz'))['img']
        self.imgMin = 0.001
        self.imgMax = 0.013

    def test_get_uint8_image(self):
        ''' Shall scale image values from float (or any) to 0 - 255 [uint8] '''

        imageUint8 = get_uint8_image(self.img1, self.imgMin, self.imgMax)

        plt.imsave(os.path.join(self.testdata_path, '0_imgUint8.png'),
                   imageUint8, vmin=0, vmax=255)

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
        keyPoints1, descr1 = find_key_points(img1)
        keyPoints2, descr2 = find_key_points(img2)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

    def test_lstsq_filter(self):
        ''' Shall filter out not matching points '''
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1)
        keyPoints2, descr2 = find_key_points(img2)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

        goodPixels = lstsq_filter(x1, y1, x2, y2)
        self.assertTrue(len(goodPixels) > len(goodPixels[goodPixels]))

if __name__ == '__main__':
    unittest.main()
