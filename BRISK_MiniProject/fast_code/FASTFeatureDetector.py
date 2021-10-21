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
        nr_of_circle_pixels = len(self._offsets)

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
        circle_pixel_status = [0]*nr_of_circle_pixels  # by default, all are invalid
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

        # Find the first change in the status
        first_change = 0
        found_change = False
        for idx in range(nr_of_circle_pixels):
            if circle_pixel_status[idx] != circle_pixel_status[(idx+1) % nr_of_circle_pixels]:
                first_change = (idx+1) % nr_of_circle_pixels
                found_change = True
                break

        if not found_change:
            # Either all are brighter or darker. Sweet, we can stop now.
            return True

        # Check to see if there are at least _min_nr_contiguous pixels of the same status.
        # Note: circle_pixel_status[] will be treated as a ring as opposed to an array.
        count_array = [0]*nr_of_circle_pixels
        count_checked_elements = 0
        max_contiguous = 0
        idx = first_change
        previous_status = circle_pixel_status[idx]
        count_array[idx] = 1
        while count_checked_elements < nr_of_circle_pixels:
            next_idx = (idx+1) % nr_of_circle_pixels
            current_status = circle_pixel_status[next_idx]
            if previous_status == current_status:
                count_array[next_idx] = count_array[idx] + 1
            else:
                if previous_status != 0 and count_array[idx] > max_contiguous:
                    max_contiguous = count_array[idx]
                previous_status = current_status
                count_array[next_idx] = 1

            idx = (idx+1) % nr_of_circle_pixels
            count_checked_elements = count_checked_elements + 1

        # Final check. Are there more enough contiguous pixels?
        if max_contiguous >= self._min_nr_contiguous:
            return True
        else:
            return False

    def getFeatures(self, img):
        keypoints = []

        lines, cols = img.shape
        print("lines "+str(lines)+ " columns " + str(cols))
        #for l in range(lines):
        #    for c in range(cols):
        #        img[l][c] = img[l][c] - 1
        #print("Done")
        for l in range(lines-6):
            print ("line "+ str(l))
            for c in range(cols-6):
                # Since the circle around the center pixel is of radius 3, then
                # the iteration has to start at index [3][3]
                ll = l+3
                cc = c+3
                # TODO: add here the quick elimination criteria. The one where we use offsets 1,5,13.
                pixel_circle = []
                for offset in self._offsets:
                    try:
                        pixel_circle.append(img[ll+offset[0]][cc+offset[1]])
                    except:
                        print("crashed with ll " + str(ll))
                        print("crashed with cc " + str(cc))
                        print("crashed with offset[0] " + str(offset[0]))
                        print("crashed with offset[1] " + str(offset[1]))
                        break
                if self._isvalidKeypoint(img[ll][cc], pixel_circle):
                    keypoints.append((ll, cc))

        return keypoints

