#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <math.h>
#include "RobBrisk.h"

#include "./fast_cpp/FASTcpp/FASTcpp/Rob7FAST.h"

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
	int threshold = 25;
	bool nms = true;
	int scales = 9;
	layerkpmat.clear();
	Ptr<Rob7FAST> rob7_fast = new Rob7FAST(threshold, true);

	for (int i = 0; i < scales; i++)
	{
		vector<vector<KeyPoint>> kpMat(layers[i].rows, vector<KeyPoint>(layers[i].cols, KeyPoint()));
		layerkpmat.push_back(kpMat);

		vector<KeyPoint> kpLayer;
		kpLayer.clear();
		//FAST(layers[i], kpLayer, threshold, nms, FastFeatureDetector::DetectorType::TYPE_9_16);
		rob7_fast->getKeypoints(layers[i], kpLayer);
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
	if (oct) //This is an octave
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
	else //This is an intraoctave
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
	
	for (int i = x - 1; i <= x + 1; i++)
	{
		for (int j = y - 1; j <= y + 1; j++)
		{
			
			if (layerkpmat[layerindex][i][j].response > max_score)
			{
				max_score = layerkpmat[layerindex][i][j].response;
				cout << max_score << endl;
			}
		}
	}
	return max_score;
}

KeyPoint ROB_Brisk::maxscoreparabola(int max_above, int max_below, int max_mid, KeyPoint kp, int scale, bool oct)
{
	/*
	The parabola is for getting the 3 max_scores of the keypoint (up mid down) fitted into a parabola
	The parabola is, supposedly a Log2 (look scientific paper)
		-> 3 sores => good_kp
		-> 3 scales
		int maxup
		int maxdown
		int maxinlayer
		int scale = i => scale of the layer => int layerup = i+1 => int layerdown = i-1
		if img or oct or intra
			<<<<< Y axis>>>>>
			scale for img (i = 0) is Log2(1)
			scale for an octave (i = even) is Log2(2^(i))
			scale for an intra (i = odd) is Log2(2^(i)*1.5
					[This gives us the y values for the three scores]
			<<<<< X axis >>>>>
					[The x values for the three scores are there specific score]
	*/

	

	int n;
	float* x = new float [n];
	float* y = new float [n];

	float scale_up = 0;
	float midscale = 0;
	float scale_down = 0;
	float a, b, c; //Parabola 

	if (scale == 0)
	{
		n = 2;
		//X value
		scale_up = (2^(scale + 1))*1.5; //For original img, the upscale is always an intraoctave
		midscale= 2^(scale);
		
		x[n] = (midscale, scale_up);
		y[n] = (max_mid, max_above);
		
	}
	else if (scale > 0 && scale < 8 )
	{
		n = 3;
		if (oct)
		{
			//X value
			scale_up = (2 ^ (scale + 1)) * 1.5;			//3.585 example
			midscale = 2 ^ (scale); //This is an oct	//2 example
			scale_down = (2 ^ (scale - 1)) * 1.5;		//1.585 example
			
			x[n] = (scale_down, midscale, scale_up);
			y[n] = (max_below, max_mid, max_above);
		}
		else
		{
			//X value
			scale_up = 2 ^ (scale + 1);
			midscale = 2 ^ (scale); //This is an intraoct
			scale_down = 2 ^ (scale - 1);
			
			x[n] = (scale_down, midscale, scale_up);
			y[n] = (max_below, max_mid, max_above);
		}

	}
	else
	{
		n = 2;
		midscale = 2 ^ (scale); //This is an intraoct
		scale_down = 2 ^ (scale - 1);

		x[n] = (scale_down, midscale);
		y[n] = (max_below, max_mid);

	}

	//Following this https://www.youtube.com/watch?v=oJRASrTlPdQ
	
	float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2y = 0, sum_x2 = 0, sum_x3 = 0, sum_x4 = 0;
	for (int i = 0; i < 3; i++)
	{
		sum_x = sum_x + x[i];
		sum_y = sum_y + y[i];
		sum_xy = sum_xy + x[i]*y[i];
		sum_x2y = sum_x2y + x[i] * x[i] * y[i];
		sum_x2 = sum_x2 + x[i] * x[i];
		sum_x3 = sum_x3 + x[i] * x[i] * x[i];
		sum_x4 = sum_x4 + x[i] * x[i] * x[i] * x[i];
	}
	float augmented_matrix[3][4];
	//Creating Augmented Matrix
	augmented_matrix[0][0] = sum_x2;
	augmented_matrix[0][1] = sum_x;
	augmented_matrix[0][2] = n;
	augmented_matrix[0][3] = sum_y;
	augmented_matrix[1][0] = sum_x3;
	augmented_matrix[1][1] = sum_x2;
	augmented_matrix[1][2] = sum_x;
	augmented_matrix[1][3] = sum_xy;
	augmented_matrix[2][0] = sum_x4;
	augmented_matrix[2][1] = sum_x3;
	augmented_matrix[2][2] = sum_x2;
	augmented_matrix[2][3] = sum_x2y;
	
	float ratio;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (j > i)
			{
				ratio = augmented_matrix[j][i] / augmented_matrix[i][i];
				for (int k = 0; k < n + 1; k++)
					augmented_matrix[j][k] = augmented_matrix[j][k] - (ratio * augmented_matrix[i][k]);
			}
		}
	}
	float sum;
	float* value = new float[n];
	value[n - 1] = augmented_matrix[n - 1][n] / augmented_matrix[n - 1][n - 1];

	for (int i = n - 2; i >= 0; i--)
	{
		sum = 0;
		for (int j = i + 1; j < n; j++)
			sum = sum + a[i][j] * value[j];
		value[i] = (augmented_matrix[i][n] - sum) / augmented_matrix[i][i];
	}
	value[0], value[1], value[2]; //a b and c from the parabola equation
	KeyPoint vertex;
	vertex.pt.x = (-value[1] / (2 * value[0]));
	vertex.pt.y = (4 * value[0] * value[2]) - (value[1] * value[1]) / (4 * value[0]);

	return vertex;
}

void ROB_Brisk::nms_scales()
{
	/*
	vector<KeyPoints> good_kp;
	for layer[k] in layers
		for kp in layer[k]
			kp_up = find_kp(layer[k+1])
			kp_down = find_kp(layer[k-1])
			if (kp_up && kp_down EXIST)
				if (score[kp] > (score[kp_up] && score[kp_down]))
					kp.parabola = get_score((score[kp],scale[kp]),(score[kp_up],scale[kp_up]),(score[kp_down],scale[kp_up]))
					kp.coordinated_interpolation = get_interpolated(kp_position)
					good_kp.push_back(kp)
	*/

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
					if (keypoints[i][j].response + 1 > max_above) //TODO: Remember to change this (+ 1), we cheated
					{
						good_kp.push_back(keypoints[i][j]);
						maxscoreparabola(max_above, -1, keypoints[i][j].response, good_kp[i], i, i % 2);
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
					if (keypoints[i][j].response + 1 > max_above && keypoints[i][j].response + 1 > max_below) //TODO: Remember to change this (+ 1), we cheated
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
					if (keypoints[i][j].response + 1 > max_below) //TODO: Remember to change this (+ 1), we cheated
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