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

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from nansat import Nansat, Domain, NSR

from sea_ice_drift.lib import (get_uint8_image,
                               get_displacement_km,
                               get_displacement_km_equirec,
                               get_displacement_pix,
                               get_denoised_object,
                               x2y2_interpolation_poly,
                               x2y2_interpolation_near,
                               get_n,
                               get_drift_vectors,
                               _fill_gpi)

from sea_ice_drift.ftlib import (find_key_points,
                                 get_match_coords,
                                 domain_filter,
                                 max_drift_filter,
                                 lstsq_filter,
                                 feature_tracking)

from sea_ice_drift.pmlib import (get_rotated_template,
                                 get_distance_to_nearest_keypoint,
                                 get_initial_rotation,
                                 rotate_and_match)

from sea_ice_drift.seaicedrift import SeaIceDrift

class SeaIceDriftLibTests(unittest.TestCase):
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
        self.testFiles = [str(f)
                     for f in np.array(testFiles)[np.argsort(dates)]]
        self.n1 = Nansat(self.testFiles[0])
        self.n2 = Nansat(self.testFiles[1])
        self.img1 = self.n1['sigma0_HV']
        self.img2 = self.n2['sigma0_HV']
        self.imgMin = 0.001
        self.imgMax = 0.013
        self.nFeatures = 5000

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
        u_dom, v_dom = get_displacement_km(self.n1, x1, y1, self.n2, x2, y2)
        u_equ, v_equ = get_displacement_km(self.n1, x1, y1, self.n2, x2, y2, 'equirec')

        lon1, lat1 = self.n1.transform_points(x1, y1)

        plt.quiver(lon1, lat1, u_dom, v_dom, color='k')
        plt.quiver(lon1, lat1, u_equ, v_equ, color='g')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertTrue(len(u_dom) == len(x1))
        self.assertTrue(len(u_equ) == len(x1))

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
        self.assertEqual(len(u), len(x1))

    def test_get_n(self):
        ''' Shall return Nansat and Matrix '''
        n = get_n(self.testFiles[0],
                           'sigma0_HV', 0.5, 0.001, 0.013, False, False)
        n = get_n(self.testFiles[0],
                           'sigma0_HV', 0.5, -20, -15, False, True)
        
        self.assertIsInstance(n, Nansat)
        self.assertEqual(n[1].dtype, np.uint8)
        self.assertEqual(n[1].min(), 0)
        self.assertEqual(n[1].max(), 255)

    def test_x2y2_interpolation_poly(self):
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        x2p1, y2p1 = x2y2_interpolation_poly(x1, y1, x2, y2, x1, y1, 1)
        x2p2, y2p2 = x2y2_interpolation_poly(x1, y1, x2, y2, x1, y1, 2)
        x2p3, y2p3 = x2y2_interpolation_poly(x1, y1, x2, y2, x1, y1, 3)

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
        
    def test_x2y2_interpolation_near(self):
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        x2p1, y2p1 = x2y2_interpolation_near(x1, y1, x2, y2, x1, y1)

        plt.subplot(1,2,1)
        plt.plot(x2, x2p1, '.')
        plt.subplot(1,2,2)
        plt.plot(y2, y2p1, '.')
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(x2p1), len(x1))

    def test_get_drift_vectors(self):
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        u, v, lon1, lat1, lon2, lat2 = get_drift_vectors(self.n1, x1, y1,
                                                         self.n2, x2, y2,
                                                         ll2km='equirec')

        plt.plot(lon1, lat1, '.')
        plt.plot(lon2, lat2, '.')
        plt.quiver(lon1, lat1, u, v)
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name)
        plt.close('all')
        self.assertEqual(len(u), len(x1))
        self.assertEqual(len(v), len(x1))

    def test_fill_gpi(self):
        a = np.array([[1,2,3],[1,2,3],[1,2,3]])
        gpi = (a > 2)
        b = _fill_gpi(a.shape, a[gpi].flatten(), gpi.flatten())
        self.assertEqual(a.shape, b.shape)
        

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


class SeaIceDriftPMLibTests(SeaIceDriftLibTests):
    def test_get_rotated_template(self):
        ''' Shall plot two templates with and without rotation '''
        temp_rot00 = get_rotated_template(self.img1, 100, 300, 50, 0)
        temp_rot10 = get_rotated_template(self.img1, 100, 300, 50, 30)

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
        img1 = get_uint8_image(self.img1, self.imgMin, self.imgMax)
        img2 = get_uint8_image(self.img2, self.imgMin, self.imgMax)
        keyPoints1, descr1 = find_key_points(img1, nFeatures=self.nFeatures)
        keyPoints2, descr2 = find_key_points(img2, nFeatures=self.nFeatures)
        x1, y1, x2, y2 = get_match_coords(keyPoints1, descr1,
                                          keyPoints2, descr2)
        
        dist = get_distance_to_nearest_keypoint(x1, y1, img1.shape)
        plt.imsave('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,
                    dist)
        self.assertEqual(dist.shape, img1.shape)

    def test_get_initial_rotation(self):
        ''' Shall find angle between images '''
        alpha12 = get_initial_rotation(self.n1, self.n2)
        alpha21 = get_initial_rotation(self.n2, self.n1)

        self.assertIsInstance(alpha12, float)
        self.assertAlmostEqual(alpha12, -alpha21, 1)
        self.assertAlmostEqual(alpha12, 60.91682335, 1)
    
    def test_rotate_and_match(self):
        ''' shall rotate and match'''
        n1 = get_n(self.testFiles[0])
        n2 = get_n(self.testFiles[1])
        (best_r, best_a, dx, dy,
         best_result, best_template) = rotate_and_match(
                         n1[1],300,100,50,n2[1],60,[-2,-1,0,1,2])
        plt.subplot(1,3,1)
        plt.imshow(n2[1], interpolation='nearest')
        plt.subplot(1,3,2)
        plt.imshow(best_result, interpolation='nearest', vmin=0)
        plt.subplot(1,3,3)
        plt.imshow(best_template, interpolation='nearest')
        plt.suptitle('%f  %f %f  %f' % (best_r, best_a, dx, dy))
        plt.savefig('sea_ice_drift_tests_%s.png' % inspect.currentframe().f_code.co_name,)
        plt.close('all')
        
        
class SeaIceDriftClassTests(SeaIceDriftLibTests):
    def test_integrated(self):
        ''' Shall use all developed functions for feature tracking'''
        lon1pm, lat1pm = np.meshgrid(np.linspace(-3, 2, 50),
                             np.linspace(86.4, 86.8, 50))

        sid = SeaIceDrift(self.testFiles[0], self.testFiles[1])
        uft, vft, lon1ft, lat1ft, lon2ft, lat2ft = sid.get_drift_FT()
        upm, vpm, rpm, apm, lon2pm, lat2pm = sid.get_drift_PM(
                                            lon1pm, lat1pm,
                                            lon1ft, lat1ft,
                                            lon2ft, lat2ft)

        lon1, lat1 = sid.n1.get_border()
        lon2, lat2 = sid.n2.get_border()
        sid.n1.reproject(Domain(NSR().wkt, '-te -3 86.4 2 86.8 -ts 500 500'))
        s01 = sid.n1['sigma0_HV']
        sid.n2.reproject(Domain(NSR().wkt, '-te -3 86.4 2 86.8 -ts 500 500'))
        s02 = sid.n2['sigma0_HV']

        plt.imshow(s01, extent=[-3, 2, 86.4, 86.8], cmap='gray', aspect=12)
        plt.quiver(lon1ft, lat1ft, uft, vft, color='r')
        plt.plot(lon2, lat2, '.-r')
        plt.xlim([-3, 2])
        plt.ylim([86.4, 86.8])
        plt.savefig('sea_ice_drift_tests_%s_img1_ft.png' % inspect.currentframe().f_code.co_name,
                    dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        plt.imshow(s02, extent=[-3, 2, 86.4, 86.8], cmap='gray', aspect=12)
        gpi = rpm > 0.4
        plt.quiver(lon1pm[gpi], lat1pm[gpi], upm[gpi], vpm[gpi], rpm[gpi])
        plt.plot(lon1, lat1, '.-r')
        plt.xlim([-3, 2])
        plt.ylim([86.4, 86.8])
        plt.savefig('sea_ice_drift_tests_%s_img2_pm.png' % inspect.currentframe().f_code.co_name,
                    dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()

