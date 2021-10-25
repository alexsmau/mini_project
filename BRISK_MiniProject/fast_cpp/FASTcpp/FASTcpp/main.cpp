#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include "Rob7FAST.h"

using namespace cv;
using namespace std;


int main()
{
	Mat image = imread("IMG_20201231_194210.jpg", IMREAD_GRAYSCALE);

	// Check for failure
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}
	/*
	for (int i = 3; i < (image.rows - 3); i++)
	{
		for (int j = 3; j < (image.cols - 3); j++)
		{
			if ((i % 100 == 0) && (j % 100 == 0))
			{
				cout << (int)(image.at<unsigned char>(i, j)) << " ";
			}
		}
		if (i % 100 == 0)
			cout << "\n";
	}
	*/

	//cout <<(int)(image.at<unsigned char>(3, 3) )<< " ";
	/*
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(25, false, FastFeatureDetector::TYPE_9_16);
	vector<KeyPoint> kpts;
	detector->detect(image, kpts);
	int count_points = kpts.size();
	cout << "OpenCV found: " << count_points << " keypoints \n";
	*/

	
	vector<KeyPoint> kpts2;
	Ptr<Rob7FAST> rob7_fast = new Rob7FAST(25, false);
	rob7_fast->getKeypoints(image, kpts2);
	//cout << "Test that the class works " << rob7_fast->offsets[0][0]<<"\n";
	
	
	/*
	String windowName = "My Image"; //Name of the window
	namedWindow(windowName, WINDOW_NORMAL); // Create a window
	imshow(windowName, image); // Show our image inside the created window.
	waitKey(0); // Wait for any keystroke in the window
	destroyWindow(windowName); //destroy the created window
	*/
	return 0;
}