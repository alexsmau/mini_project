#pragma once

#ifndef ROB7_BRISK_H
#define ROB7_BRISK_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>

#include <iostream>

using namespace cv;
using namespace std;

/*
*	main.cpp
*
*	ROB_Brisk brisk1 = ROB_Brisk(image);
*	vector<Descriptors>  descriptors = brisk1.get_descriptors();
* 
*  RobBrisk
*  get_descriptors()
*		1.- Create scale-space Pyramid (Layers are divided into Octaves and Intra-octaves)
		2.- Compute FAST across scale-space (scores)
		3.- Pixel level non-maximal suppression
		4.- Compute sub-pixel maximum across patch
		5.- Re-interpolate image coordinates from scale-space



*/


class ROB_Brisk
{

public:
	ROB_Brisk(Mat img);
	void descriptors(); //Function for generating the Descriptor
	

private:
	Mat image; //Variable for the image

	//Feature Detection
	vector<Mat> layers[8]; //Vector containing the downsampled images
	vector<KeyPoint> keypoints[8]; //Vector containing the keypoijnts obtained from FAST

	void create_scale_space(); //Function for "creating" the Scale-Space pyramid
	void computeFAST(); //Function for computing the FAST algorithm, returning the keypoints found
	void nms_scales(); //Function for computing non-maximum supression between layers
	void compute_subpixel_maximum(); //Function for computing subpixel maximum
	void reinterpolate(); //Function for re-interpolating the image coordinates

	//Feature Description
	void sampling(); //Function for sampling points around keypoint
	void pair_generation(); //Function for generating pairs of points
	void pair_division(); //Function for dividing the pairs found into Long and Short
	void distance_computation(); //Function for computing the overall distance of the keypoint
};

#endif