#include "Rob7BriskDescriptor.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <stdio.h>

using namespace std;

Rob7BriskDescriptor::Rob7BriskDescriptor(KeyPoint kp)
{
	nr_of_circles = 4;

	keypoint.pt.x = kp.pt.x;
	keypoint.pt.y = kp.pt.y;
	keypoint.response = kp.response;
	keypoint.size = kp.size;

	nr_of_points_per_circle[0] = 10;
	nr_of_points_per_circle[1] = 14;
	nr_of_points_per_circle[2] = 15;
	nr_of_points_per_circle[3] = 20;
	circle_pixel_radius[0] = 4;
	circle_pixel_radius[1] = 7;
	circle_pixel_radius[2] = 11;
	circle_pixel_radius[3] = 15;

	for (int i = 0; i < 16; i++)
	{
		descriptor[i] = 0;
	}
}

void Rob7BriskDescriptor::createDescriptor(Mat image)
{
	generateOffsets();
	generatePoints();
	compute_short_and_long_pairs();
	double orientation = compute_orientation(image);
	for (int circle = 0; circle < nr_of_circles; circle++)
	{
		circle_offsets[circle] += orientation;
	}
	generatePoints();
	compute_descriptor_bits(image);
}

static void print_debug_mat(int debug_mat[41][41])
{
	for (int i = 0; i < 41; i++)
	{
		for (int j = 0; j < 41; j++)
		{
			cout << debug_mat[i][j] << " ";
		}
		cout << " i = " << i << "\n";
	}
}

void Rob7BriskDescriptor::generatePoints()
{
	points[0][0] = keypoint.pt.y; //row in image matrix
	points[0][1] = keypoint.pt.x; //column in image matrix
	int pos = 1;
	float angle_increment;
	int dx, dy;
	for (int circle = 0; circle < nr_of_circles; circle++)
	{
		angle_increment = (2 * M_PI) / nr_of_points_per_circle[circle];
		for (int point = 0; point < nr_of_points_per_circle[circle]; point++)
		{
			dx = (int)(circle_pixel_radius[circle] * cos(circle_offsets[circle] + point * angle_increment));
			dy = (int)(circle_pixel_radius[circle] * sin(circle_offsets[circle] + point * angle_increment));

			points[pos][0] = points[0][0] + dx;
			points[pos][1] = points[0][1] + dy;
			pos++;
		}
		distance_per_circle[circle] = sqrt((points[pos - 1][0] - points[pos - 2][0]) * (points[pos - 1][0] - points[pos - 2][0]) + (points[pos - 1][1] - points[pos - 2][1]) * (points[pos - 1][1] - points[pos - 2][1]));
	}

}

void Rob7BriskDescriptor::generateOffsets()
{
	for (int i = 0; i < nr_of_circles; i++)
	{
		circle_offsets[i] = 2 * M_PI * rand();
	}
}

double Rob7BriskDescriptor::gauss(double sigma, double x)
{
	double expVal = -1 * (pow(x, 2) / pow(2 * sigma, 2));
	double divider = sqrt(2 * M_PI * pow(sigma, 2));
	return (1 / divider) * exp(expVal);
}

double Rob7BriskDescriptor::calculateDistance(int point1_idx, int point2_idx)
{
	int dx = points[point1_idx][0] - points[point2_idx][0];
	int dy = points[point1_idx][1] - points[point2_idx][1];
	return sqrt(dx * dx + dy * dy);
}

void Rob7BriskDescriptor::compute_short_and_long_pairs()
{
	int short_pair_idx = 0;
	int long_pair_idx = 0;
	double distance;
	double max_thr = 9.75;
	double min_thr = 13.00;
	for (int i = 0; i < 60; i++)
	{
		for (int j = i + 1; j < 60; j++)
		{
			distance = calculateDistance(i, j);
			if (distance < max_thr && short_pair_idx < 512)
			{
				short_pairs[short_pair_idx][0] = i;
				short_pairs[short_pair_idx][1] = j;
				short_pair_idx++;
			}
			else if (distance > min_thr && long_pair_idx < 870)
			{
				long_pairs[long_pair_idx][0] = i;
				long_pairs[long_pair_idx][1] = j;
				long_pair_idx++;
			}
		}
	}
	count_long_pairs = long_pair_idx;
}

double Rob7BriskDescriptor::compute_orientation(Mat img)
{
	double gx, gy;
	double cma_gx = 0;
	double cma_gy = 0;
	for (int i = 0; i < count_long_pairs; i++)
	{
		calculate_gardient(long_pairs[i][0], long_pairs[i][1], &gx, &gy, img);
		cma_gx = (gx + i*cma_gx)/(i+1);
		cma_gy = (gy + i*cma_gy)/(i+1);
	}

	return atan2(cma_gy, cma_gx);
}

void Rob7BriskDescriptor::calculate_gardient(int point1_idx, int point2_idx, double* gx, double* gy, Mat image)
{
	double sigma_gain = 10000000000;
	double distance = calculateDistance(point1_idx, point2_idx);
	//double intensity1 = gauss(get_point_sigma(point1_idx) * sigma_gain, ((double)(image.at<unsigned char>(points[point1_idx][0], points[point1_idx][1]))));
	//double intensity2 = gauss(get_point_sigma(point2_idx) * sigma_gain, ((double)(image.at<unsigned char>(points[point2_idx][0], points[point2_idx][1]))));
	double intensity1 = ((double)(image.at<unsigned char>(points[point1_idx][0], points[point1_idx][1])));
	double intensity2 = ((double)(image.at<unsigned char>(points[point2_idx][0], points[point2_idx][1])));
	double magnitude = (intensity1 - intensity2) / distance;
	
	*gx = (points[point1_idx][0] - points[point2_idx][0]) * magnitude;
	*gy = (points[point1_idx][1] - points[point2_idx][1]) * magnitude;

}

double Rob7BriskDescriptor::get_point_sigma(int point_idx)
{
	if (1 >= point_idx && point_idx <= 10)
		return distance_per_circle[0];

	if (11 >= point_idx && point_idx <= 24)
		return distance_per_circle[1];

	if (25 >= point_idx && point_idx <= 39)
		return distance_per_circle[2];

	return distance_per_circle[3];
}

void Rob7BriskDescriptor::compute_descriptor_bits(Mat image)
{
	double intensity_j, intensity_i;
	int img_r_i, img_c_i, img_r_j, img_c_j;
	for (int i = 0; i < 512; i++)
	{
		img_r_i = points[short_pairs[i][0]][0];
		img_c_i = points[short_pairs[i][0]][1];
		img_r_j = points[short_pairs[i][1]][0];
		img_c_j = points[short_pairs[i][1]][1];
		intensity_i = ((double)(image.at<unsigned char>(img_r_i, img_c_i)));
		intensity_j = ((double)(image.at<unsigned char>(img_r_j, img_c_j)));

		if (intensity_j > intensity_i)
		{
			descriptor[i / 32] |= (1 << (i % 32));
		}
	}
}

int Rob7BriskDescriptor::compareTo(Rob7BriskDescriptor desc)
{
	uint32_t xor_val;
	int count = 0;
	for (int i = 0; i < 16; i++)
	{
		xor_val = descriptor[i] ^ desc.descriptor[i];
		if (xor_val)
		{
			count++;
		}
		while (xor_val = (xor_val & (xor_val - 1)))
		{
			count++;
		}
	}
	return count;
}