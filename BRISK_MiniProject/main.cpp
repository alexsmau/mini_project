/*
	BRISK(general)
	-> Feature Detection
		1.- Create scale-space Pyramid (Layers are divided into Octaves and Intra-octaves)
		2.- Compute FAST across scale-space (scores)
		3.- Pixel level non-maximal suppression
		4.- Compute sub-pixel maximum across patch
		5.- Re-interpolate image coordinates from scale-space

	-> Feature Description
		1.- Sampling around keypoint
		2.- Generate pairs of points
		3.- Divide into Short- and Long-distance pairs
		4.- Use Long-distance pairs to compute overall distance of the keypoint
		5.- Generate the Descriptor
*/

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "RobBrisk.h"

using namespace cv;
using namespace std;


void cheatBrisk(vector<Mat> images)
{
	vector<KeyPoint> keypointsA, keypointsB;
	Mat descriptorsA, descriptorsB;
	
	int Threshl = 100;
	int Octaves = 4; //(pyramid layer) from which the keypoint has been extracted
	float PatternScales = 1.0f;
	Ptr<Feature2D> detector = BRISK::create(Threshl, Octaves, PatternScales);

	detector->detect(images[4], keypointsA);
	detector->detect(images[5], keypointsB);
	detector->compute(images[4], keypointsA, descriptorsA);
	detector->compute(images[5], keypointsB, descriptorsB);

	vector<DMatch> matches;
	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
	matcher.match(descriptorsA, descriptorsB, matches);

	Mat all_matches;
	drawMatches(images[4], keypointsA, images[5], keypointsB, matches, all_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("BRISK All Matches", all_matches);
	waitKey(0);
}

int main()
{	

	Mat image = imread("C://Users//migno//Desktop//Robotics//ROB7//Project//Git//mini_project//BRISK_MiniProject//images//Dog.jpg", IMREAD_GRAYSCALE);
	ROB_Brisk brisk1 = ROB_Brisk(image);

	// 
	//string load_img_folder = "C://Users//migno//Desktop//Robotics//ROB7//Project//Git//mini_project//BRISK_MiniProject//images";
	//string save_img = load_img_folder + "result";

	
	brisk1.descriptors();
	

	/*
	for (int j = 0; j < 9; j++)
	{
		imshow("Result", layers[j]);
		waitKey(0);
	}
	*/

	//cheatBrisk(orderedimg);
}