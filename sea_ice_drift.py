import numpy as np
import cv2


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
