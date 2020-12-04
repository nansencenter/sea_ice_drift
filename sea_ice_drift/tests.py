# Name:    tests.py
# Purpose: Unit tests
# Authors:      Anton Korosov
# Created:      26.08.2016
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

import os
import sys
import glob
import unittest
import inspect
from mock import MagicMock, patch

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from nansat import Nansat, Domain, NSR

from sea_ice_drift.lib import (get_uint8_image,
                               get_displacement_km,
                               get_displacement_pix,
                               get_denoised_object,
                               interpolation_poly,
                               interpolation_near,
                               get_n,
                               get_drift_vectors,
                               _fill_gpi,
                              get_invalid_mask)

from sea_ice_drift.ftlib import (find_key_points,
                                 get_match_coords,
                                 domain_filter,
                                 max_drift_filter,
                                 lstsq_filter,
                                 feature_tracking)

from sea_ice_drift.pmlib import (get_template,
                                 get_distance_to_nearest_keypoint,
                                 get_initial_rotation,
                                 rotate_and_match)

from sea_ice_drift.seaicedrift import SeaIceDrift

class SeaIceDriftTestBase(unittest.TestCase):
    def setUp(self):
        ''' Load test data '''
        testDir = os.getenv('ICE_DRIFT_TEST_DATA_DIR')
        if testDir is None:
            sys.exit('ICE_DRIFT_TEST_DATA_DIR is not defined')
        testFiles = glob.glob(os.path.join(testDir, 'S1?_*tif'))
        if len(testFiles) < 2:
            sys.exit('Not enough test files in %s' % testDir)
        # sort by date
        dates = [os.path.basename(f).split('_')[4] for f in testFiles]
        self.testFiles = [str(f) for f in np.array(testFiles)[np.argsort(dates)]]
        self.n1 = Nansat(self.testFiles[0])
        self.n2 = Nansat(self.testFiles[1])
        self.imgMin = 0.001
        self.imgMax = 0.013
        self.nFeatures = 5000
        self.img1 = get_uint8_image(self.n1['sigma0_HV'], self.imgMin, self.imgMax, 0, 0)
        self.img2 = get_uint8_image(self.n2['sigma0_HV'], self.imgMin, self.imgMax, 0, 0)

class SeaIceDriftLibTests(SeaIceDriftTestBase):
    def test_get_uint8_image(self):
        ''' Shall scale image values from float (or any) to 0 - 255 [uint8] '''
        imageUint8 = get_uint8_image(self.n1['sigma0_HV'], self.imgMin, self.imgMax, 0, 0)
        plt.imsave('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,
                    imageUint8, vmin=0, vmax=255)

        self.assertEqual(imageUint8.dtype, np.uint8)
        self.assertEqual(imageUint8.min(), 1)
        self.assertEqual(imageUint8.max(), 255)

    def test_get_displacement_km(self):
        ''' Shall find matching coordinates and plot quiver in lon/lat'''
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        h = get_displacement_km(self.n1, x1, y1, self.n2, x2, y2)

        plt.scatter(x1, y1, 30, h)
        plt.colorbar()
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertTrue(len(h) == len(x1))

    def test_get_displacement_pix(self):
        ''' Shall find matching coordinates and plot quiver in pixel/line'''
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        u, v = get_displacement_pix(self.n1, x1, y1, self.n2, x2, y2)

        plt.quiver(x1, y1, u, v)
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(u), len(x1))

    @patch('sea_ice_drift.lib.get_invalid_mask')
    def test_get_n(self, mock_get_invalid_mask):
        invalid_mask =  np.zeros((500,500)).astype(bool)
        invalid_mask[:100,:100] = True
        mock_get_invalid_mask.return_value = invalid_mask
        n = get_n(self.testFiles[0],
                           bandName='sigma0_HV',
                           factor=0.5,
                           vmin=0.001,
                           vmax=0.013,
                           denoise=False,
                           dB=False)
        n = get_n(self.testFiles[0],
                           bandName='sigma0_HV',
                           factor=0.5,
                           vmin=-20,
                           vmax=-15,
                           denoise=False,
                           dB=True)

        self.assertIsInstance(n, Nansat)
        self.assertEqual(n[1].dtype, np.uint8)
        self.assertEqual(n[1].min(), 0)
        self.assertEqual(n[1].max(), 255)

    def test_get_invalid_mask_all_valid(self):
        n = Nansat(self.testFiles[0])
        img = n[1]
        mask = np.zeros((np.array(img.shape)/20).astype(int))
        n.watermask = MagicMock(return_value=[None, mask])
        mask = get_invalid_mask(img, n, 20)
        self.assertEqual(np.where(mask)[0].size, 0)

    def test_get_invalid_mask_some_valid(self):
        n = Nansat(self.testFiles[0])
        img = n[1]
        mask = np.zeros((np.array(img.shape)/20).astype(int))
        mask[:10,:] = 2
        n.watermask = MagicMock(return_value=[None, mask])
        mask = get_invalid_mask(img, n, 20)
        self.assertGreater(np.where(mask)[0].size, 0)

    def test_get_invalid_mask_with_error(self):
        n = Nansat(self.testFiles[0])
        n.watermask = MagicMock(return_value=None, side_effect=KeyError('foo'))
        img = n[1]
        mask = get_invalid_mask(img, n, 20)
        self.assertFalse(np.any(mask))

    def test_interpolation_poly(self):
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        x2p1, y2p1 = interpolation_poly(x1, y1, x2, y2, x1, y1, 1)
        x2p2, y2p2 = interpolation_poly(x1, y1, x2, y2, x1, y1, 2)
        x2p3, y2p3 = interpolation_poly(x1, y1, x2, y2, x1, y1, 3)

        plt.subplot(1,2,1)
        plt.plot(x2, x2p1, '.')
        plt.plot(x2, x2p2, '.')
        plt.plot(x2, x2p3, '.')
        plt.subplot(1,2,2)
        plt.plot(y2, y2p1, '.')
        plt.plot(y2, y2p2, '.')
        plt.plot(y2, y2p3, '.')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(x2p1), len(x1))

    def test_interpolation_near(self):
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        x2p1, y2p1 = interpolation_near(x1, y1, x2, y2, x1, y1)

        plt.subplot(1,2,1)
        plt.plot(x2, x2p1, '.')
        plt.subplot(1,2,2)
        plt.plot(y2, y2p1, '.')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(x2p1), len(x1))

    def test_get_drift_vectors(self):
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        u, v, lon1, lat1, lon2, lat2 = get_drift_vectors(self.n1, x1, y1,
                                                   self.n2, x2, y2)

        plt.plot(lon1, lat1, '.')
        plt.plot(lon2, lat2, '.')
        plt.quiver(lon1, lat1, u, v, angles='xy', scale_units='xy', scale=0.25)
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(u), len(x1))
        self.assertEqual(len(v), len(x1))

    def test_fill_gpi(self):
        a = np.array([[1,2,3],[1,2,3],[1,2,3]])
        gpi = (a > 2)
        b = _fill_gpi(a.shape, gpi.flatten(), a[gpi].flatten())
        self.assertEqual(a.shape, b.shape)


class SeaIceDriftFTLibTests(SeaIceDriftTestBase):
    def setUp(self):
        super(SeaIceDriftFTLibTests, self).setUp()
        self.keyPoints1, self.descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        self.keyPoints2, self.descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)

    def test_find_key_points(self):
        ''' Shall find key points using default values '''
        self.assertTrue(len(self.keyPoints1) > 1000)

    def test_get_match_coords(self):
        ''' Shall find matching coordinates '''
        x1, y1, x2, y2 = get_match_coords(self.keyPoints1, self.descr1,
                                          self.keyPoints2, self.descr2)
        self.assertTrue(len(self.keyPoints1) > len(x1))
        self.assertTrue(len(self.keyPoints2) > len(x2))

    def test_domain_filter(self):
        ''' Shall leave keypoints from second image withn domain of the first '''
        self.keyPoints2f, self.descr2f = domain_filter(self.n2, self.keyPoints2, self.descr2,
                                             self.n1, domainMargin=100)

        # plot dots
        cols1 = [kp.pt[0] for kp in self.keyPoints1]
        rows1 = [kp.pt[1] for kp in self.keyPoints1]
        lon1, lat1 = self.n1.transform_points(cols1, rows1, 0)
        cols2 = [kp.pt[0] for kp in self.keyPoints2f]
        rows2 = [kp.pt[1] for kp in self.keyPoints2f]
        lon2, lat2 = self.n2.transform_points(cols2, rows2, 0)
        plt.plot(lon1, lat1, '.')
        plt.plot(lon2, lat2, '.')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')

        self.assertTrue(len(self.descr2f) < len(self.descr2))

    def test_max_drift_filter_speed(self):
        '''Shall keep only slow drift '''
        x1, y1, x2, y2 = get_match_coords(self.keyPoints1, self.descr1,
                                          self.keyPoints2, self.descr2,
                                          ratio_test=0.7)
        x1f, y1f, x2f, y2f = max_drift_filter(self.n1, x1, y1,
                                          self.n2, x2, y2,
                                          max_speed=0.001)
        self.assertGreater(len(x1f), 0)
        self.assertGreater(len(x1), len(x1f))

        # remove time_coverage_start
        self.n1.vrt.dataset.SetMetadata({})

        # test that
        with self.assertRaises(ValueError):
            x1f, y1f, x2f, y2f = max_drift_filter(self.n1, x1, y1,
                                          self.n2, x2, y2,
                                          max_speed=0.3)


        x1f, y1f, x2f, y2f = max_drift_filter(self.n1, x1, y1,
                                          self.n2, x2, y2,
                                          max_drift=100)
        self.assertGreater(len(x1f), 0)
        self.assertGreater(len(x1), len(x1f))

    def test_lstsq_filter(self):
        ''' Shall filter out not matching points '''
        x1, y1, x2, y2 = get_match_coords(self.keyPoints1, self.descr1,
                                          self.keyPoints2, self.descr2,
                                          ratio_test=0.8)

        x1f, y1f, x2f, y2f = lstsq_filter(x1, y1, x2, y2)
        self.assertTrue(len(x1) > len(x1f))


class SeaIceDriftPMLibTests(SeaIceDriftTestBase):
    def test_get_template(self):
        ''' Shall plot two templates with and without rotation '''
        temp_rot00 = get_template(self.img1, 100, 300, 0, 50)
        temp_rot10 = get_template(self.img1, 100, 300, 30, 50)

        plt.subplot(1,2,1)
        plt.imshow(temp_rot00, interpolation='nearest')
        plt.subplot(1,2,2)
        plt.imshow(temp_rot10, interpolation='nearest')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')

        self.assertEqual(temp_rot00.shape, (50,50))
        self.assertEqual(temp_rot10.shape, (50,50))

    def test_get_distance_to_nearest_keypoint(self):
        ''' Shall create vector of distances '''
        keyPoints1, descr1 = find_key_points(self.img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(self.img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)

        dist = get_distance_to_nearest_keypoint(x1, y1, self.img1.shape)
        plt.imsave('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,
                    dist)
        self.assertEqual(dist.shape, self.img1.shape)

    def test_get_initial_rotation(self):
        ''' Shall find angle between images '''
        alpha12 = get_initial_rotation(self.n1, self.n2)
        alpha21 = get_initial_rotation(self.n2, self.n1)

        self.assertIsInstance(alpha12, float)
        self.assertAlmostEqual(np.floor(alpha12), np.floor(-alpha21), 1)
        self.assertAlmostEqual(alpha12, -3.85, 1)

    def test_rotate_and_match(self):
        ''' shall rotate and match'''
        n1 = get_n(self.testFiles[0])
        n2 = get_n(self.testFiles[1])
        dx, dy, best_a, best_r, best_h, best_result, best_template = rotate_and_match(
                         n1[1], 300, 100, 50, n2[1], 0, [-3,-2,-1,0,1,2,3])
        plt.subplot(1,3,1)
        plt.imshow(n2[1], interpolation='nearest')
        plt.subplot(1,3,2)
        plt.imshow(best_result, interpolation='nearest', vmin=0)
        plt.subplot(1,3,3)
        plt.imshow(best_template, interpolation='nearest')
        plt.suptitle('%f %f %f %f %f' % (dx, dy, best_a, best_r, best_h))
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,)
        plt.close('all')


class SeaIceDriftClassTests(SeaIceDriftTestBase):
    @patch('sea_ice_drift.lib.get_invalid_mask')
    def test_integrated(self, mock_get_invalid_mask):
        ''' Shall use all developed functions for feature tracking'''

        invalid_mask =  np.zeros((500,500)).astype(bool)
        invalid_mask[:100,:100] = True
        mock_get_invalid_mask.return_value = invalid_mask

        sid = SeaIceDrift(self.testFiles[0], self.testFiles[1])

        lon1b, lat1b = sid.n1.get_border()
        lon1pm, lat1pm = np.meshgrid(np.linspace(lon1b.min(), lon1b.max(), 50),
                             np.linspace(lat1b.min(), lat1b.max(), 50))
        uft, vft, lon1ft, lat1ft, lon2ft, lat2ft = sid.get_drift_FT()
        upm, vpm, apm, rpm, hpm, lon2pm, lat2pm = sid.get_drift_PM(
                                            lon1pm, lat1pm,
                                            lon1ft, lat1ft,
                                            lon2ft, lat2ft)

        lon1, lat1 = sid.n1.get_border()
        lon2, lat2 = sid.n2.get_border()
        ext_str = '-te %s %s %s %s -ts 500 500' % (lon1b.min(), lat1b.min(), lon1b.max(), lat1b.max())
        sid.n1.reproject(Domain(NSR().wkt, ext_str))
        s01 = sid.n1['sigma0_HV']
        sid.n2.reproject(Domain(NSR().wkt, ext_str))
        s02 = sid.n2['sigma0_HV']
        extent=[lon1b.min(), lon1b.max(), lat1b.min(), lat1b.max()]
        plt.imshow(s01, extent=extent, cmap='gray', aspect=10)
        plt.quiver(lon1ft, lat1ft, uft, vft, color='r',
                   angles='xy', scale_units='xy', scale=0.2)
        plt.plot(lon2, lat2, '.-r')
        plt.xlim([lon1b.min(), lon1b.max()])
        plt.ylim([lat1b.min(), lat1b.max()])
        plt.savefig('sea_ice_drift_tests_%s_img1_ft.png' % inspect.currentframe().f_code.co_name,
                    dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        plt.imshow(s02, extent=extent, cmap='gray', aspect=10)
        gpi = hpm*rpm > 4
        plt.quiver(lon1pm[gpi], lat1pm[gpi], upm[gpi], vpm[gpi], rpm[gpi]*hpm[gpi],
                   angles='xy', scale_units='xy', scale=0.2)
        plt.plot(lon1, lat1, '.-r')
        plt.xlim([lon1b.min(), lon1b.max()])
        plt.ylim([lat1b.min(), lat1b.max()])
        plt.savefig('sea_ice_drift_tests_%s_img2_pm.png' % inspect.currentframe().f_code.co_name,
                    dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()

