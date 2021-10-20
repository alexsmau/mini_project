"""
This is the ROB7-763 group's implementation of the Features from Accelerated Segment Test algorithm.

This file contains one class called FASTFeatureDetector
"""

class FASTFeatureDetector:
    def __init__(self, threshold = 10, non_max_suppression = False):
        self._t = threshold
        self._nms = non_max_suppression
        slef._offsets = [(), (), (), (), ]

    def setThreshold(self, threshold):
        self._t = threshold

    def setNonMaxSuppression(self, nms):
        self._nms = nms

    def getFeatures(self, img):


