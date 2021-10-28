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


void compare(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2)
{
	bool exists;
	int count = 0;
	for (KeyPoint kp1 : kpts1)
	{
		exists = false;
		for (KeyPoint kp2 : kpts2)
		{
			if ((kp1.pt.x == kp2.pt.x) && (kp1.pt.y == kp2.pt.y))
			{
				exists = true;
				count++;
				break;
			}
		}

		if (!exists && (count == 2))
		{
			cout << "Keypoint at col " << kp1.pt.x << " and row " << kp1.pt.y;
			return;
		}
		exists = false;
	}

}

void compare2(vector<KeyPoint> kpts1, vector<KeyPoint> kpts2)
{	
	int minD, distance;
	for (KeyPoint kp1 : kpts1)
	{
		minD = 10000;
		for (KeyPoint kp2 : kpts2)
		{
			distance = (kp1.pt.x > kp2.pt.x) ? (kp1.pt.x - kp2.pt.x) : (kp2.pt.x - kp1.pt.x);
			distance += (kp1.pt.y > kp2.pt.y) ? (kp1.pt.y - kp2.pt.y) : (kp2.pt.y - kp1.pt.y);

			if (distance < minD)
			{
				minD = distance;
			}
		}
		cout << "dist: " << minD << "\n";
	}
}

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
	
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(25, true, FastFeatureDetector::TYPE_9_16);
	vector<KeyPoint> kpts;
	detector->detect(image, kpts);
	int count_points = kpts.size();
	cout << "OpenCV found: " << count_points << " keypoints \n";
	
	//Mat blur_img;
	//GaussianBlur(image, blur_img, Size(3, 3), 0, 0);
	vector<KeyPoint> kpts2;
	Ptr<Rob7FAST> rob7_fast = new Rob7FAST(25, true);
	rob7_fast->getKeypoints(image, kpts2);
	cout << "Rob7 Fast found: " << kpts2.size() << " keypoints \n";

	bool exists;
	int count = 0;
	for (KeyPoint rob_kp : kpts2)
	{
		exists = false;
		for (KeyPoint cv_kp : kpts)
		{
			if ((rob_kp.pt.x == cv_kp.pt.x) && (rob_kp.pt.y == cv_kp.pt.y))
			{
				exists = true;
				//cout << "rob score: " << rob_kp.response << " cv score " << cv_kp.response << "\n";
				break;
			}
		}

		if (!exists)
		{
			count++;
		}
	}
	cout << "There are " << count << " kp found by rob but not by opencv\n";

	compare(kpts, kpts2);
	compare2(kpts2, kpts);
	
	/*
	String windowName = "My Image"; //Name of the window
	namedWindow(windowName, WINDOW_NORMAL); // Create a window
	imshow(windowName, image); // Show our image inside the created window.
	waitKey(0); // Wait for any keystroke in the window
	destroyWindow(windowName); //destroy the created window
	*/
	return 0;
}