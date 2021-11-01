#pragma once

#ifndef ROB7_BRISK_H
#define ROB7_BRISK_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "Rob7BriskDescriptor.h"

using namespace cv;
using namespace std;


class ROB_Brisk
{

public:
	ROB_Brisk(Mat img);
	void calculate_descriptors(); //Function for generating the Descriptor
	
	//temporary public.
	void gaussEliminationLS(int nr_of_coeff, int n, double augMat[5][5], double coeff[5]);
	/**
	 * Function which computed the best-fit quadratic parabola given 3 points.
	 * [in]  x[3]     - and array conatining all the x values: x1, x2, x3
	 * [in]  y[3]     - and array conatining all the y values: y1, y2, y3
	 * [out] coeff[5] - an array which will be populated with the quadratic coefficients
	 *                - If quadratic has this form: a * x^2 + b * x + c, then
	 *                - c = coeff[0], b = coeff[1], a = coeff[2], Note the reverse order!
	 */
	void computeQuadraticCoeff(double x[3], double y[3], double coeff[5]);

	vector<Rob7BriskDescriptor> descriptors;
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
	int getmaxscoreinarea(int layerindex, int x, int y, bool up, bool oct, float* x_pos, float* y_pos); //Function for getting the maximum score of a 3x3 area
	

	/*
	 * Compute the maximum point of the parabola that best fits the 3 keypoint locations. 
	 * [in]  scores[3]       - an array of all the scores scores[0]-below, score[1]-middle, score[2]-above.
	 * [in]  mid_point_layer - the layer of the midpoint. From this we can calculate the scaleof all points.
	 * [out] max_point_layer - the layer in which the max point of the parabola is. This is log2(scale of the max point).
	 * [out] max_point_score - the score of the maximum point of the parabola.
	 */
	void max_score_form_parabola(double scores[3], int mid_point_layer, double* max_point_layer, double* max_point_score);

	/*
	 * We need to figure out where in the original image does this keypoint belong to.
	 * The x and y coordinates are the same ones as the keypoint, but the layer in which it resides 
	 * is no longer "i" but rather the sub-layer computed from the best-fit parabola. From the (x/y) pair and 
	 * layer, the corresponding (x,y) in the original image.
	 * 
	 * [in]  kp_row     - the row in the layer for which keypoint corresponds to
	 * [in]  kp_col     - the col in the layer for which keypoint corresponds to
	 * [in]  layer      - the layer in which the kp now is. Note: this will be a real number not an integer!!!
	 * [out] image_row  - the row in the original image that the keypoint corresponds to 
	 * [out] image_col  - the column in the original image that the keypoint corresponds to 
	 */
	void extrapolate_kp_location_in_image(int kp_row, int kp_col, float positions[4], int i, double layer, double* image_row, double* image_col);
	void compute_subpixel_maximum(); //Function for computing subpixel maximum
	void reinterpolate(); //Function for re-interpolating the image coordinates

	//Feature Description
	void generate_descriptors_form_keypoints();
};

#endif