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
	layerkpmat.clear();

	for (int i = 0; i < scales; i++)
	{
		vector<vector<KeyPoint>> kpMat(layers[i].rows, vector<KeyPoint>(layers[i].cols, KeyPoint()));
		layerkpmat.push_back(kpMat);

		vector<KeyPoint> kpLayer;
		kpLayer.clear();
		FAST(layers[i], kpLayer, threshold, nms, FastFeatureDetector::DetectorType::TYPE_9_16);
		keypoints.push_back(kpLayer);

		for (KeyPoint kp : kpLayer)
		{
			kpMat[kp.pt.y][kp.pt.x].pt.x = kp.pt.x;
			kpMat[kp.pt.y][kp.pt.x].pt.y = kp.pt.y;
			if (kp.response <= 0)
			{
				kpMat[kp.pt.y][kp.pt.x].response = 1;
				kp.response = 1;
			}
			else
			{
				kpMat[kp.pt.y][kp.pt.x].response = kp.response;
			}
		}
	}
}

int ROB_Brisk::getmaxscoreinarea(int layerindex, int x, int y, bool up, bool oct)
{
	if (oct)// this is an octave
	{
		if (up)
		{
			x = 2 * x / 3;
			y = 2 * y / 3;
		}
		else
		{
			x = 4 * x / 3;
			y = 4 * y / 3;
		}
	}
	else //this is an intraoctave
	{
		if (up)
		{
			x = 3 * x / 4;
			y = 3 * y / 4;
		}
		else
		{
			x = 3 * x / 2;
			y = 3 * y / 2;
		}
	}

	int max_score = 0;
	cout << x << " " << y << endl;
	for (int i = x - 1; i <= x + 1; i++)
	{
		for (int j = y - 1; j <= y + 1; j++)
		{
			cout << max_score << endl;
			if (layerkpmat[layerindex][i][j].response > max_score)
			{
				max_score = layerkpmat[layerindex][i][j].response;
			}
		}
	}
	return max_score;
}

void ROB_Brisk::nms_scales()
{
	/*
	PROBLEMS:
	- We need to have access to the scores of all the keypoints found by FAST (Alex?¿)
	- Does there exist a function that computes NMS between scales?
	- We need an easy way of NMS2.0, compare the keypoints on adyacent scales and remove the smaller ones
	- - sort()

	vector<KeyPoints> good_kp;
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

	vector<KeyPoint> downscaled;
	vector<KeyPoint> upscaled;

	for (int i = 0; i < layerkpmat.size(); i++)
	{
		int max_above = 0;
		int max_below = 0;

		if (i == 0)
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				max_above = getmaxscoreinarea(i + 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, true, i % 2 ? false : true);
				if (max_above > 0) //Checks if they exist
				{
					if (keypoints[i][j].response + 1 > max_above)
					{
						good_kp.push_back(keypoints[i][j]);
						cout << "We compute the parabola" << endl;
					}
				}
			}
		}
		else if (i > 0 && i < 8)
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				max_above = getmaxscoreinarea(i + 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, true, i % 2 ? false : true);
				max_below = getmaxscoreinarea(i - 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, false, i % 2 ? false : true);
				if (max_above > 0 && max_below > 0) //Checks if they exist
				{
					if (keypoints[i][j].response + 1 > max_above && keypoints[i][j].response + 1 > max_below) //TODO: Remember to change this, we cheated
					{
						good_kp.push_back(keypoints[i][j]);
						cout << "We compute the parabola" << endl;
					}
				}
			}
		}
		else
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				max_below = getmaxscoreinarea(i - 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, false, i % 2 ? false : true);
				if (max_below > 0) //Checks if they exist
				{
					if (keypoints[i][j].response + 1 > max_below)
					{
						good_kp.push_back(keypoints[i][j]);
						cout << "We compute the parabola" << endl;
					}
				}
			}
		}
	}
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