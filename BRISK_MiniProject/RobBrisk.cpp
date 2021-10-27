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

ROB_Brisk::ROB_Brisk(Mat img) 
{
	image = img.clone();
}

//Keypoint Detection
void ROB_Brisk::create_scale_space()
{
	vector<vector<Mat>> octaves;
	vector<vector<Mat>> intraoctaves;

	int scales = 8;

	vector<Mat> img_;
	Mat octave;
	vector<Mat> _img;
	Mat intraoctave;

	for (int j = 1; j < scales; j = j + 2)
	{
		if (octave.empty())
		{	
			layers.push_back(image);
			resize(image, intraoctave, Size(2 * (image.cols) / 3, 2 * (image.rows) / 3), INTER_LINEAR);
			layers.push_back(intraoctave);
			pyrDown(image, octave, Size((image.cols) / 2, (image.rows) / 2));
			layers.push_back(octave);	
		}
		else
		{
			pyrDown(intraoctave, intraoctave, Size((intraoctave.cols) / 2, (intraoctave.rows) / 2));
			layers.push_back(intraoctave);
			pyrDown(octave, octave, Size((octave.cols) / 2, (octave.rows) / 2));
			layers.push_back(octave);
		}
	}
}

void ROB_Brisk::computeFAST() 
{
	int threshold = 50;
	bool nms = true;
	int scales = 9;
	vector<KeyPoint> keypoint;
	int col = layer[i].cols;
	int row = image.rows;

	int Matrix[row][col];

	//= Mat::zeros(Size(image.cols, image.rows), CV_8U);

	for (int i = 0; i < scales; i++)
	{
		FAST(layers[i], keypoint, threshold, nms, FastFeatureDetector::DetectorType::TYPE_9_16);
		keypoints.push_back(keypoint);

		for (int j = 0; j < keypoint.size(); j++)
		{
			//Matrix[keypoint[j].pt.x][keypoint[j].pt.y] = 1;

		}
		//Matrix[keypoint.pt.x][keypoint.pt.y] = keypoint[score]
	}
}



void ROB_Brisk::nms_scales() 
{
	/*
	vector<KeyPoints> good_kp;
>>>>>>> Adding what I forgot
	for layer[k] in layers
		for kp in layer[k]
			kp_up = find_kp(layer[k+1])
			kp_down = find_kp(layer[k-1])
			if (kp_up && kp_down EXIST)
				if (score[kp] > (score[kp_up] && score[kp_down]))
					kp.parabola = get_score((score[kp],score[kp]),(score[kp_up],scale[kp_up]),(score[kp_down],scale[kp_up]))
					kp.coordinated_interpolation = get_interpolated(kp_position)
					good_kp.push_back(kp)
	*/
	
	vector<vector<KeyPoint>> good_kp;

	vector<KeyPoint> downscaled;
	vector<KeyPoint> upscaled;

	int k = i + 1;
	if (k < img.size())
	{
		downscaled = keypoints[k];
	}
	int m = i - 1;
	if (m >= 0)
	{
		upscaled = keypoints[m];
	}

		/*
			PROBLEMS:
			- We need to have access to the scores of all the keypoints found by FAST (Alex?¿)
			- Does there exist a function that computes NMS between scales?
			- We need an easy way of NMS2.0, compare the keypoints on adyacent scales and remove the smaller ones
			- - sort()
		*/

}



void ROB_Brisk::compute_subpixel_maximum() {}
void ROB_Brisk::reinterpolate() {}

//Keypoint Description
void ROB_Brisk::sampling() {}
void ROB_Brisk::pair_generation() {}
void ROB_Brisk::pair_division() {}
void ROB_Brisk::distance_computation() {}

void ROB_Brisk::descriptors() 
{
	create_scale_space();
	computeFAST();
	nms_scales();

}