#pragma once

#ifndef ROB7_BRISK_H
#define ROB7_BRISK_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>

#include <iostream>

using namespace cv;
using namespace std;

class ROB_Brisk
{

public:
	ROB_Brisk(Mat img);
	void descriptor(); //Function for generating the Descriptor
	

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