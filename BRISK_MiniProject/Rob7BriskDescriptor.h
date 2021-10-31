#pragma once

#ifndef ROB7_BRISKDESCRIPTOR_H
#define ROB7_BRISKDESCRIPTOR_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

class Rob7BriskDescriptor
{
private:
	KeyPoint keypoint;
	int nr_of_circles;
	double circle_offsets[4];
	double nr_of_points_per_circle[4];
	double distance_per_circle[4];
	int circle_pixel_radius[4];
	int points[60][2];
	int long_pairs[870][2];
	int short_pairs[512][2];
	int count_long_pairs;
	int _scale;
	uint32_t descriptor[16];
	void generatePoints();
	void generateOffsets();
	double gauss(double sigma, double x);
	double calculateDistance(int point1_idx, int point2_idx);
	void compute_short_and_long_pairs();
	void calculate_gardient(int point1_idx, int point2_idx, double *gx, double *gy, Mat image);
	double get_point_sigma(int point_idx);
	double compute_orientation(Mat img);
public:
	Rob7BriskDescriptor(KeyPoint kp, int scale);
	void createDescriptor(Mat image);
};

#endif
