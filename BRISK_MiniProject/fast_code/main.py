import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import FASTFeatureDetector as ffd

def open_src_fast():
    print ("Start code")
    img = cv.imread('IMG_20201231_194210.jpg',cv.IMREAD_GRAYSCALE)

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()  # FastFeatureDetector()

    # find and draw the keypoints
    thresh = 25
    fast.setThreshold(thresh)
    fast.setNonmaxSuppression(0)

    kp = fast.detect(img, None)

    # This is our code
    rob7_fast = ffd.FASTFeatureDetector()

    key_pts = rob7_fast.getFeatures(img)

    print ("shape of image rob7 key points " + str(key_pts.shape))


    print ("shape of image is " +str(img.shape))
    print("type " + str(type(img[0][0])))

    print ("max "+ str(np.iinfo(img[0][0].dtype).max))

    print ("There are " + str(len(kp)) + " keypoints")

    print (str (kp[0].pt))
    print(str(kp[0].size))
    print(str(kp[0].response))

    print(str(kp[100].pt))
    print(str(kp[100].size))
    print(str(kp[100].response))

    img2 = cv.drawKeypoints(img, kp, img, color=(255, 0, 0))
    plt.imshow(img2), plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    open_src_fast()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
