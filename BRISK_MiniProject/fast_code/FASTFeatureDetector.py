"""
This is the ROB7-763 group's implementation of the Features from Accelerated Segment Test algorithm.

This file contains one class called FASTFeatureDetector
"""
import numpy as np

class FASTFeatureDetector:
    def __init__(self, threshold = 10, non_max_suppression = False):
        self._min_nr_contiguous = 12
        self._t = threshold
        self._nms = non_max_suppression
        self._offsets = [(-3,  0), #1
                         (-3,  1), #2
                         (-2,  2), #3
                         (-1,  3), #4
                         ( 0,  3), #5
                         ( 1,  3), #6
                         ( 2,  2), #7
                         ( 3,  1), #8
                         ( 3,  0), #9
                         ( 3, -1), #10
                         ( 2, -2), #11
                         ( 1, -3), #12
                         ( 0, -3), #13
                         (-1, -3), #14
                         (-2, -2), #15
                         (-3, -1)] #16

    def setThreshold(self, threshold):
        self._t = threshold

    def setNonMaxSuppression(self, nms):
        self._nms = nms

    def _isvalidKeypoint(self, center_pixel, circle_of_pixels):
        nr_of_circle_pixels = self._offsets.size()

        # The max number of pixels on the circle that can be invalid (between the center pixel's intensity plus
        # or minus the threshold value)
        max_nr_invalid = nr_of_circle_pixels - self._min_nr_contiguous

        type_max_value = np.iinfo(center_pixel.dtype).max
        type_min_value = np.iinfo(center_pixel.dtype).min

        if (center_pixel + self._t) > type_max_value:
            Ihigh = type_max_value
        else:
            Ihigh = center_pixel + self._t

        if (center_pixel - self._t) < type_min_value:
            Ilow = type_min_value
        else:
            Ilow = center_pixel - self._t

        # Create a 1D array of size nr_of_circle_pixels that will indicate the relationship
        # between intensity of each pixel on the circle an the center pixel's intensity.
        # 0  - Invalid: pixel is between [center_intensity - threshold, center_intensity + threshold]
        # 1  - Brighter than center_intensity + threshold
        # -1 - Darker than center_intensity - threshold
        circle_pixel_status = [0]*nr_of_circle_pixels # by default, all are invalid
        invalid_count = 0
        for i in range(nr_of_circle_pixels):
            if circle_of_pixels[i] > Ihigh:
                circle_pixel_status[i] = 1
            elif circle_of_pixels[i] < Ilow:
                circle_pixel_status[i] = -1
            else:
                invalid_count = invalid_count + 1

            if invalid_count > max_nr_invalid:
                # No point in continuing.
                return False

    def getFeatures(self, img):
        keypoints = []

        lines, cols = img.shape
        for l in range(lines-3):
            for c in range(cols-3):
                # Since the circle around the center pixel is of radius 3, then
                # the iteration has to start at index [3][3]
                ll = l+3
                cc = c+3
                #TODO: add here the quick elimination criteria. The one where we use offsets 1,5,13.
                pixel_circle = []
                for offset in self._offsets:
                    pixel_circle.append(img[ll+offset[0]][cc+offset[1]])
                if self._isvalidKeypoint(img[ll][cc], pixel_circle):
                    keypoints.append((ll, cc))

