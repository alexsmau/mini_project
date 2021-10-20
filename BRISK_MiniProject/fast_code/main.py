import numpy as np
import cv2
from matplotlib import pyplot as plt

def open_src_fast():
    print ("Start code")
    img = cv2.imread('IMG_20201231_194210.jpg',0)
    print ("Marker 1")
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector()
    print ("Marker 2")
    # find and draw the keypoints
    kp = fast.detect(img, None)
    print ("Marker 3")
    img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
    print ("Marker 4")
    # Print all default params
    print ("Threshold: ", fast.getInt('threshold'))
    print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
    print ("neighborhood: ", fast.getInt('type'))
    print ("Total Keypoints with nonmaxSuppression: ", len(kp))

    cv2.imwrite('fast_true.png',img2)

    # Disable nonmaxSuppression
    fast.setBool('nonmaxSuppression',0)
    kp = fast.detect(img,None)

    print ("Total Keypoints without nonmaxSuppression: ", len(kp))

    img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

    cv2.imwrite('fast_false.png',img3)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    open_src_fast()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
