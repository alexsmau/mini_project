#include "Rob7BriskDescriptor.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>

using namespace std;

Rob7BriskDescriptor::Rob7BriskDescriptor(KeyPoint kp, int scale)
{
	_scale = scale;

	nr_of_circles = 4;

	keypoint.pt.x = kp.pt.x;
	keypoint.pt.y = kp.pt.y;
	keypoint.response = kp.response;

	nr_of_points_per_circle[0] = 10;
	nr_of_points_per_circle[1] = 14;
	nr_of_points_per_circle[2] = 15;
	nr_of_points_per_circle[3] = 20;
	circle_pixel_radius[0] = 4;
	circle_pixel_radius[1] = 7;
	circle_pixel_radius[2] = 11;
	circle_pixel_radius[3] = 15;

}

void Rob7BriskDescriptor::createDescriptor(Mat image)
{
	generateOffsets();
	generatePoints();
	compute_short_and_long_pairs();
	double orientation = compute_orientation(image);
	cout << "orientation is: " << orientation<<"\n";
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
	/*
	int debug_mat[41][41];
	for (int i = 0; i < 41; i++)
		for (int j = 0; j < 41; j++)
			debug_mat[i][j] = 0;
	*/

	points[0][0] = keypoint.pt.y; //row in image matrix
	points[0][1] = keypoint.pt.y; //column in image matrix
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

			//debug_mat[21 + dx][21 + dy] = 1;
			//cout << "dx " << dx << " dy " << dy << "\n";
			points[pos][0] = points[0][0] + dx;
			points[pos][1] = points[0][1] + dy;
			pos++;
		}
		//cout << "circle " << circle << "\n";
		//print_debug_mat(debug_mat);
		distance_per_circle[circle] = sqrt((points[pos - 1][0] - points[pos - 2][0]) * (points[pos - 1][0] - points[pos - 2][0]) + (points[pos - 1][1] - points[pos - 2][1]) * (points[pos - 1][1] - points[pos - 2][1]));
	}
	/*
	for (int circle = 0; circle < nr_of_circles; circle++)
	{
		cout << "circle " << circle << " has a distance of: " << distance_per_circle[circle] << "\n";
	}
	*/
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
	cout << "There are " << short_pair_idx << " short pairs idx\n";
	cout << "There are " << long_pair_idx << " long pairs idx\n";
}

double Rob7BriskDescriptor::compute_orientation(Mat img)
{
	double gx, gy;
	double average_gx = 0;
	double average_gy = 0;
	for (int i = 0; i < count_long_pairs; i++)
	{
		calculate_gardient(long_pairs[i][0], long_pairs[i][1], &gx, &gy, img);
		average_gx += gx;
		average_gy += gy;
	}
	average_gx /= count_long_pairs;
	average_gy /= count_long_pairs;

	return atan2(average_gy, average_gx);
}

void Rob7BriskDescriptor::calculate_gardient(int point1_idx, int point2_idx, double* gx, double* gy, Mat image)
{
	double distance = calculateDistance(point1_idx, point2_idx);
	double intensity1 = gauss(get_point_sigma(point1_idx), ((double)(image.at<unsigned char>(points[point1_idx][0], points[point1_idx][1]))));
	double intensity2 = gauss(get_point_sigma(point2_idx), ((double)(image.at<unsigned char>(points[point2_idx][0], points[point2_idx][1]))));
	double magnitude = (intensity1 - intensity2) / distance;
	*gx = (points[point1_idx][0] - points[point2_idx][0]) / magnitude;
	*gy = (points[point1_idx][1] - points[point2_idx][1]) / magnitude;
}

double Rob7BriskDescriptor::get_point_sigma(int point_idx)
{
	if (1 >= point_idx && point_idx <= 10)
		return distance_per_circle[0];

	if (11 >= point_idx && point_idx <= 24)
		return distance_per_circle[1];

	if (25 >= point_idx && point_idx <= 39)
		return distance_per_circle[2];

	if (40 >= point_idx && point_idx <= 59)
		return distance_per_circle[3];
}