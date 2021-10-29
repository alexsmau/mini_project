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
	void descriptors(); //Function for generating the Descriptor
	

private:
	Mat image; //Variable for the image

	
	vector<Mat> layers; //Vector containing the downsampled images
	vector<vector<KeyPoint>> keypoints; //Vector containing the keypoijnts obtained from FAST
	vector<vector<vector<KeyPoint>>> layerkpmat;
	vector<KeyPoint> good_kp; //

	//Feature Detection
	void create_scale_space(); //Function for "creating" the Scale-Space pyramid
	void computeFAST(); //Function for computing the FAST algorithm, returning the keypoints found
	void nms_scales(); //Function for computing non-maximum supression between layers
	int getmaxscoreinarea(int layerindex, int x, int y, bool up, bool oct); //Function for getting the maximum score of a 3x3 area
	KeyPoint maxscoreparabola(int max_above, int max_below, int max_mid, KeyPoint kp, int scale, bool oct); //Function for computing the max score of a keypoint based on logarithmic parabola
	void compute_subpixel_maximum(); //Function for computing subpixel maximum
	void reinterpolate(); //Function for re-interpolating the image coordinates

	//Feature Description
	void sampling(); //Function for sampling points around keypoint
	void pair_generation(); //Function for generating pairs of points
	void pair_division(); //Function for dividing the pairs found into Long and Short
	void distance_computation(); //Function for computing the overall distance of the keypoint
};

#endif