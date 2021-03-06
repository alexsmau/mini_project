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

#define PRESENT_STEPS

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
#ifdef PRESENT_STEPS
		//namedWindow("Octave", WINDOW_NORMAL);
		imshow("Octave", octave);
		//namedWindow("Intraoctave", WINDOW_NORMAL);
		imshow("Intraoctave", intraoctave);
		waitKey(0);
#endif
	}
	int count = 0;
#ifdef PRESENT_STEPS
	destroyWindow("Octave");
	destroyWindow("Intraoctave");
#endif
}

void ROB_Brisk::computeFAST()
{
	int threshold = 10;
	bool nms = true;
	int scales = 9;
	layerkpmat.clear();
	Ptr<Rob7FAST> rob7_fast = new Rob7FAST(threshold, true);

	for (int i = 0; i < scales; i++)
	{
		vector<vector<KeyPoint>> kpMat(layers[i].rows, vector<KeyPoint>(layers[i].cols, KeyPoint()));
		vector<KeyPoint> kpLayer;
		kpLayer.clear();

		rob7_fast->getKeypoints(layers[i], kpLayer);
		keypoints.push_back(kpLayer);
#ifdef PRESENT_STEPS
		Mat layer_w_kp;
		drawKeypoints(layers[i], kpLayer, layer_w_kp);
		imshow("Layer with FAST keypoints", layer_w_kp);
		waitKey(0);
		destroyWindow("Layer with FAST keypoints");
#endif // PRESENT_STEPS

		for (KeyPoint kp : kpLayer)
		{
			kpMat[kp.pt.y][kp.pt.x].pt.x = kp.pt.x;
			kpMat[kp.pt.y][kp.pt.x].pt.y = kp.pt.y;
			if (kp.response <= 0)
			{
				cout << "Kp at x: " << kp.pt.x << " y: " << kp.pt.y << " has response: " << kp.response << "\n";
				kpMat[kp.pt.y][kp.pt.x].response = 1;
				kp.response = 1;
			}
			else
			{
				kpMat[kp.pt.y][kp.pt.x].response = kp.response;
			}
		}
		layerkpmat.push_back(kpMat);
	}

}

int ROB_Brisk::getmaxscoreinarea(int layerindex, int x, int y, bool up, bool oct, float* x_pos, float* y_pos)
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
				*x_pos = layerkpmat[layerindex][i][j].pt.y;
				*y_pos = layerkpmat[layerindex][i][j].pt.x;
			}
		}
	}
	return max_score;
}

/*******
Function adapted from: https://www.bragitoff.com/2018/06/polynomial-fitting-c-program/
********/
void ROB_Brisk::gaussEliminationLS(int nr_of_coeff, int n, double augMat[5][5], double coeff[5]) {
	int i, j, k;
	for (i = 0; i < nr_of_coeff - 1; i++) {
		//Partial Pivoting
		for (k = i + 1; k < nr_of_coeff; k++) {
			//If diagonal element(absolute vallue) is smaller than any of the terms below it
			if (fabs(augMat[i][i]) < fabs(augMat[k][i])) {
				//Swap the rows
				for (j = 0; j < n; j++) {
					double temp;
					temp = augMat[i][j];
					augMat[i][j] = augMat[k][j];
					augMat[k][j] = temp;
				}
			}
		}
		//Begin Gauss Elimination
		for (k = i + 1; k < nr_of_coeff; k++) {
			double  term = augMat[k][i] / augMat[i][i];
			for (j = 0; j < n; j++) {
				augMat[k][j] = augMat[k][j] - term * augMat[i][j];
			}
		}

	}
	//Begin Back-substitution
	for (i = nr_of_coeff - 1; i >= 0; i--) {
		coeff[i] = augMat[i][n - 1];
		for (j = i + 1; j < n - 1; j++) {
			coeff[i] = coeff[i] - augMat[i][j] * coeff[j];
		}
		coeff[i] = coeff[i] / augMat[i][i];
	}
}
#if 0
static void printMatrix(int m, int n, double matrix[5][5])
{
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("%lf\t", matrix[i][j]);
		}
		printf("\n");
	}
}
#endif
/*******
Function adapted from: https://www.bragitoff.com/2018/06/polynomial-fitting-c-program/
********/
void ROB_Brisk::computeQuadraticCoeff(double x[3], double y[3], double coeff[5])
{
	//no. of data-points
	int nr_of_data_pt = 3;
	//degree of polynomial
	const int pol_deg = 2;

	int i, j;
	// an array of size 2*n+1 for storing N, Sig xi, Sig xi^2, ...., etc. which are the independent components of the normal matrix
	double X[2 * pol_deg + 1];
	for (i = 0; i <= 2 * pol_deg; i++) {
		X[i] = 0;
		for (j = 0; j < nr_of_data_pt; j++) {
			X[i] = X[i] + pow(x[j], i);
		}
	}
	//the normal augmented matrix
	double augMat[5][5];
	// rhs
	double Y[pol_deg + 1];
	for (i = 0; i <= pol_deg; i++) {
		Y[i] = 0;
		for (j = 0; j < nr_of_data_pt; j++) {
			Y[i] = Y[i] + pow(x[j], i) * y[j];
		}
	}
	for (i = 0; i <= pol_deg; i++) {
		for (j = 0; j <= pol_deg; j++) {
			augMat[i][j] = X[i + j];
		}
	}
	for (i = 0; i <= pol_deg; i++) {
		augMat[i][pol_deg + 1] = Y[i];
	}

	gaussEliminationLS(pol_deg + 1, pol_deg + 2, augMat, coeff);

}

void ROB_Brisk::max_score_form_parabola(double scores[3], int mid_point_layer, double* max_point_layer, double* max_point_score)
{
	/* The prabola will have the scale as the x-axis and the scores as the y-axis. */
	double x[3];
	//Asumption is that mid_point_layer will be greater or equal to 1;
	//Bottom kp scale
	x[0] = pow(2, (mid_point_layer - 1));
	if ((mid_point_layer - 1) % 2) //if it is an intraoctave 
	{
		x[0] *= 1.5;
	}
	// Middle kp scale
	x[1] = pow(2, mid_point_layer);
	if (mid_point_layer % 2) //if it is an intraoctave 
	{
		x[1] *= 1.5;
	}
	// Above kp scale
	x[2] = pow(2, (mid_point_layer + 1));
	if ((mid_point_layer + 1) % 2) //if it is an intraoctave 
	{
		x[2] *= 1.5;
	}
	double coeff[5];

	computeQuadraticCoeff(x, scores, coeff);

	//Find the peak of the parabola
	double a, b, c;
	a = coeff[2];
	b = coeff[1];
	c = coeff[0];

	double xp; // x-coordinate of the peak
	xp = (-1 * b) / (2 * a);
	double yp; // y-coordinate of the peak
	yp = a * (xp * xp) + b * xp + c;

	//remember that the scale is exponential, and we are interested in the layer.
	*max_point_layer = log2(xp);
	*max_point_score = yp;
}

void ROB_Brisk::extrapolate_kp_location_in_image(int kp_row, int kp_col, float positions[4], int i, double layer, float* image_row, float* image_col)
{
	double scale, scale_going_through_currentlayer, scale_NOT_going_through_currentlayer;
	bool scale_up = false;

	/*
	* octave going up = 2/3
	* octave going down = 4/3
	* intraoctave going up = 3/4
	* intraoctave going down = 3/2
	*/

	if (i % 2) //current layer = intraoctave
	{
		if (layer > i)
		{
			scale = 0.75;
			scale_going_through_currentlayer = 0.66; //(we go from layer below, which is an octave, to current layer and then to layer)
			scale_NOT_going_through_currentlayer = 1.33;
			scale_up = true;
		}
		else //layer < i
		{
			scale = 1.5;
			scale_going_through_currentlayer = 1.33; //(we go from layer above, which is an octave, to current layer and then to layer)
			scale_NOT_going_through_currentlayer = 0.66;
			scale_up = false;
		}
	}
	else //current layer = octave 
	{
		if (layer > i)
		{
			scale = 0.66;
			scale_going_through_currentlayer = 0.75; //(we go from layer below, which is an intraoctave, to current layer and then to layer)
			scale_NOT_going_through_currentlayer = 1.5;
			scale_up = true;
		}
		else //(current layer = octave && layer < i)
		{
			scale = 1.33;
			scale_going_through_currentlayer = 1.5; //(we go from layer above, which is an intraoctave, to current layer and then to layer)
			scale_NOT_going_through_currentlayer = 0.75;
			scale_up = false;
		}
	}

	double whole;
	double decimal_part = modf(layer, &whole);
	double layer_value_middle, layer_value_lower, layer_value_upper;

	layer_value_middle  = 1 - ((1 - scale) * (decimal_part));
	if (scale_up == true)
	{
		layer_value_lower = scale_going_through_currentlayer * (1 - ((1 - scale) * (decimal_part)));
		layer_value_upper = 1 - ((1 - scale_NOT_going_through_currentlayer) * (1 - decimal_part));
	}
	else
	{
		layer_value_lower = 1 - ((1 - scale_NOT_going_through_currentlayer) * (1 - decimal_part));
		layer_value_upper = scale_going_through_currentlayer * (1 - ((1 - scale) * (decimal_part)));
	}

	*image_row = (positions[0] * layer_value_upper + positions[2] * layer_value_lower + kp_row * layer_value_middle) / 3; //Average x pos
	*image_col = (positions[1] * layer_value_upper + positions[3] * layer_value_lower + kp_col * layer_value_middle) / 3; //Average y pos
}

void ROB_Brisk::nms_scales()
{
	for (int i = 0; i < layerkpmat.size(); i++)
	{
		int max_above = 0;
		int max_below = 0;

		if (i == 0)
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				float x_pos_above, y_pos_above;
				max_above = getmaxscoreinarea(i + 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, true, i % 2 ? false : true, &x_pos_above, &y_pos_above);
				if (max_above > 0) //Checks if they exist
				{
					if (keypoints[i][j].response + 1 > max_above) //TODO: Remember to change this (+ 1), we cheated
					{
						//good_kp.push_back(keypoints[i][j]);
						//maxscoreparabola(max_above, -1, keypoints[i][j].response, good_kp[i], i, i % 2);
						//cout << "At layer 0 the current score is: " << keypoints[i][j].response << " and the above is: " << max_above << "\n";
					}
				}
			}
		}
		else if (i > 0 && i < 8)
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				float x_pos_above, y_pos_above, x_pos_below, y_pos_below;
				max_above = getmaxscoreinarea(i + 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, true, i % 2 ? false : true, &x_pos_above, &y_pos_above);
				max_below = getmaxscoreinarea(i - 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, false, i % 2 ? false : true, &x_pos_below, &y_pos_below);
				float positions[4];
				positions[0] = x_pos_above;
				positions[1] = y_pos_above;
				positions[2] = x_pos_below;
				positions[3] = y_pos_below;

				if (max_above > 0 && max_below > 0) //Checks if they exist
				{
					if (keypoints[i][j].response > max_above && keypoints[i][j].response > max_below)
					{
						double scores[3];
						scores[0] = (double)max_below;
						scores[1] = (double)keypoints[i][j].response;
						scores[2] = (double)max_above;
						double max_point_layer, max_point_score;
						max_score_form_parabola(scores, i, &max_point_layer, &max_point_score);
						float img_row, img_col;

						extrapolate_kp_location_in_image(keypoints[i][j].pt.y, keypoints[i][j].pt.x, positions, i, max_point_layer, &img_row, &img_col);
						/*
						 * I think this is what we need to do here: push to the good list a new keypoint that is found in the original image
						 * exatrpolated from the keypoint we found in this layer and the max_layer/score we found from the parabola.
						 */
						 good_kp.push_back(cv::KeyPoint(img_col, img_row, max_point_layer, 1, max_point_score, 0, -1));

					}
				}
			}
		}
		else
		{
			for (int j = 0; j < keypoints[i].size(); j++)
			{
				float x_pos_below, y_pos_below;
				max_below = getmaxscoreinarea(i - 1, keypoints[i][j].pt.y, keypoints[i][j].pt.x, false, i % 2 ? false : true, &x_pos_below, &y_pos_below);
				if (max_below > 0) //Checks if they exist
				{
					if (keypoints[i][j].response + 1 > max_below) //TODO: Remember to change this (+ 1), we cheated
					{
						//good_kp.push_back(keypoints[i][j]);
						//cout << "At layer: " << i << " the current score is : " << keypoints[i][j].response << " below is : " << max_below << "\n";
					}
				}
			}
		}
	}
	cout << "There are " << good_kp.size() << " good keypoints.\n";
}

//Keypoint Description
void ROB_Brisk::generate_descriptors_form_keypoints()
{
	for (KeyPoint kp : good_kp)
	{
		int newColsNr = image.cols / (pow(2, kp.size));
		int newRowsNr = image.rows / (pow(2, kp.size));

		if (kp.pt.x < 40 || kp.pt.x >(newColsNr - 40)) //kp column
			continue;
		if (kp.pt.y < 40 || kp.pt.y >(newRowsNr - 40)) //kp row
			continue;

		Rob7BriskDescriptor descriptor = Rob7BriskDescriptor(kp);
		Mat scaled_image;
		resize(image, scaled_image, Size(newColsNr, newRowsNr), INTER_LINEAR);

		descriptor.createDescriptor(scaled_image);
		descriptors.push_back(descriptor);
	}
}

void ROB_Brisk::calculate_descriptors()
{
	create_scale_space();
	computeFAST();
	nms_scales();
	generate_descriptors_form_keypoints();
}