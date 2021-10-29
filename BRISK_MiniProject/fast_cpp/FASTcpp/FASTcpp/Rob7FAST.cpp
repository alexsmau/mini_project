#include "Rob7FAST.h"
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

Rob7FAST::Rob7FAST(int threshold, bool non_maximum_suppression)
{
	t = threshold;
	nms = non_maximum_suppression;
	offsets[0][0] = -3;
	offsets[1][0] = -3;
	offsets[2][0] = -2;
	offsets[3][0] = -1;
	offsets[4][0] = 0;
	offsets[5][0] = 1;
	offsets[6][0] = 2;
	offsets[7][0] = 3;
	offsets[8][0] = 3;
	offsets[9][0] = 3;
	offsets[10][0] = 2;
	offsets[11][0] = 1;
	offsets[12][0] = 0;
	offsets[13][0] = -1;
	offsets[14][0] = -2;
	offsets[15][0] = -3;

	offsets[0][1] = 0;
	offsets[1][1] = 1;
	offsets[2][1] = 2;
	offsets[3][1] = 3;
	offsets[4][1] = 3;
	offsets[5][1] = 3;
	offsets[6][1] = 2;
	offsets[7][1] = 1;
	offsets[8][1] = 0;
	offsets[9][1] = -1;
	offsets[10][1] = -2;
	offsets[11][1] = -3;
	offsets[12][1] = -3;
	offsets[13][1] = -3;
	offsets[14][1] = -2;
	offsets[15][1] = -1;
	maxIntensity = 255;
	minIntensity = 0;
	min_nr_contiguous = 12;

	debug = false;
}

int Rob7FAST::getPixelScoreFromImage(cv::Mat image, int x, int y)
{
	int circle_of_pixels[16];
	int ll, cc;

	for (int k = 0; k < 16; k++)
	{
		ll = x + offsets[k][0];
		cc = y + offsets[k][1];
		circle_of_pixels[k] = (int)(image.at<unsigned char>(ll, cc));
	}

	//std::cout << "i " << i << " j " << j << " ll " << ll << " cc " << cc << "\n";
	/*
	if (i == 3 && j == 363)
	{
		for (int k = 0; k < 16; k++)
			std::cout << circle_of_pixels[k] << " ";
		std::cout << "\n center pix: " << ((int)(image.at<unsigned char>(i, j))) << "\n";
		std::cout << "\nThis is the surrpunding pix mat:\n";
		debug = true;
		for (int k = i - 3; k <= i + 3; k++)
		{
			for (int l = j - 3; l <= j + 3; l++)
			{
				std::cout << ((int)(image.at<unsigned char>(k, l))) << " ";
			}
			std::cout << "\n";
		}
	}
	debug = true;
	*/
	return getPixelScore(circle_of_pixels, ((int)(image.at<unsigned char>(x, y))));
}


void Rob7FAST::getKeypoints(cv::Mat image, std::vector<cv::KeyPoint> &keypoints)
{
	int circle_of_pixels[16];
	int ll, cc;
	int debug_count = 0;
	int fast_score;
	int max_score;
	int max_r, max_c;
	int debug_r, debug_c,debug_foo =0;

	try {
		if (nms)
		{
			int total_cols = image.cols - 6;
			int remaining_cols = total_cols % 3;
			int total_rows = image.rows - 6;
			int remaining_rows = total_rows % 3;
			for (int i = 3; i < (image.rows - 3) - remaining_rows; i += 3)
			{
				for (int j = 3; j < (image.cols - 3) - remaining_cols; j += 3)
				{
					max_score = -1;
					for (int r = 0; r < 3; r++)
					{
						for (int c = 0; c < 3; c++)
						{
							//if (i>=3996)
							//	std::cout << "Foo1 Look at r: " << i + r << " c: " << j + c << "\n";
							debug_foo = 1;
							debug_r = i + r;
							debug_c = j + c;
							fast_score = getPixelScoreFromImage(image, i + r, j + c);
							if (fast_score > max_score)
							{
								max_score = fast_score;
								max_r = i + r;
								max_c = j + c;
							}
						}
					}
					if (max_score != -1)
					{
						keypoints.push_back(cv::KeyPoint(max_c, max_r, 1, 1, max_score, 0, -1));
					}
				}
				if (remaining_cols)
				{
					max_score = -1;
					for (int r = 0; r < 3; r++)
					{
						for (int c = (total_cols - remaining_cols); c < total_cols; c++)
						{
							//std::cout << "Foo2 Look at r: " << i + r << " c: " <<  c << "\n";
							debug_foo = 2;
							debug_r = i + r;
							debug_c = c;
							fast_score = getPixelScoreFromImage(image, r + i, c);
							if (fast_score > max_score)
							{
								max_score = fast_score;
								max_r = r + i;
								max_c = c;
							}
						}
					}
					if (max_score != -1)
					{
						keypoints.push_back(cv::KeyPoint(max_c, max_r, 1, 1, max_score, 0, -1));
					}
				}
				//std::cout << "\n\nline done\n";
			}
			if (remaining_rows)
			{
				for (int j = 3; j < (image.cols - 3) - remaining_cols; j += 3)
				{
					max_score = -1;
					for (int r = (total_rows - remaining_rows); r < (total_rows); r++)
					{
						for (int c = 0; c < 3; c++)
						{
							//std::cout << "Foo3 Look at r: " << r << " c: " << j + c << "\n";
							debug_foo = 3;
							debug_r = r;
							debug_c = j + c;
							fast_score = getPixelScoreFromImage(image, r, j + c);
							if (fast_score > max_score)
							{
								max_score = fast_score;
								max_r = r;
								max_c = j + c;
							}
						}
					}
					if (max_score != -1)
					{
						keypoints.push_back(cv::KeyPoint(max_c, max_r, 1, 1, max_score, 0, -1));
					}
				}
			}
			if (remaining_cols && remaining_rows)
			{
				max_score = -1;
				for (int r = (total_rows - remaining_rows); r < (total_rows); r++)
				{
					for (int c = (total_cols - remaining_cols); c < total_cols; c++)
					{
						//std::cout << "Foo4 Look at r: " <<  r << " c: " <<  c << "\n";
						debug_foo = 4;
						debug_r = r;
						debug_c = c;
						fast_score = getPixelScoreFromImage(image, r, c);
						if (fast_score > max_score)
						{
							max_score = fast_score;
							max_r = r;
							max_c = c;
						}
					}
				}
				if (max_score != -1)
				{
					keypoints.push_back(cv::KeyPoint(max_c, max_r, 1, 1, max_score, 0, -1));
				}
			}
		}
		else
		{
			//std::cout << "rows " << image.rows << " cols " << image.cols << "\n";
			for (int i = 3; i < (image.rows - 3); i++)
			{
				for (int j = 3; j < (image.cols - 3); j++)
				{

					fast_score = getPixelScoreFromImage(image, i, j);
					debug = false;
					if (fast_score != -1)
					{
						if (fast_score > 1)
						{
							debug_count++;
						}
						/* Carefull, a matrix is using (i,j) as (row,column) but Point(x,y) is using (x,y) as (column,row).
						 * As such the Point2d that descrives this keypoint is at x = j  and y = i.
						 * It is super confusing, so be carefull when taking a keypoint and maping it to a matrix !!!
						 */
						keypoints.push_back(cv::KeyPoint(j, i, 1, 1, fast_score, 0, -1));
					}

				}
;
			}

			std::cout << " Rob7 algo has found " << debug_count << " keypoints \n";
		}
	}
	catch (...)
	{
		std::cout << "kill me\n";
		std::cout << "Died at r: " << debug_r << " c: " << debug_c << " foo " << debug_foo << "\n";
		//std::cout << "remaining_rows: " << remaining_rows << " remaining_cols: " << remaining_cols <<"\n";
		while(true){}
		
	}


}

int Rob7FAST::getPixelScore(int circle_of_pixels[16], int center_pixel)
{
	int Imax, Imin;
	int max_invalid = 16 - min_nr_contiguous;
	int invalid_count = 0;
	
	Imax = (center_pixel > (maxIntensity - t)) ? maxIntensity : (center_pixel + t);
	Imin = (center_pixel < (minIntensity + t)) ? minIntensity : (center_pixel - t);

	if (debug && false)
	{
		std::cout << "Circle of pixels\n";
		for (int i = 0; i < 16; i++)
			std::cout << circle_of_pixels[i] << " ";
		std::cout << "\nImax "<<Imax<<" Imin "<<Imin<<"\n";
		//for (int i = 0; i < 16; i++)
		//	std::cout << pixel_status[i] << " ";
	}
	/**
	 * Create a 1D array of size nr_of_circle_pixels that will indicate the relationship
	 * between intensity of each pixel on the circle an the center pixel's intensity.
	 *  0 - Invalid: pixel is between[center_intensity - threshold, center_intensity + threshold]
	 *  1 - Brighter than center_intensity + threshold
	 * -1 - Darker than center_intensity - threshold
	 */
	int pixel_status[16];
	for (int i = 0; i < 16; i++)
	{
		if (circle_of_pixels[i] > Imax)
		{
			pixel_status[i] = 1;
		}
		else if (circle_of_pixels[i] < Imin)
		{
			pixel_status[i] = -1;
		}
		else
		{
			pixel_status[i] = 0;
			invalid_count++;
			if (invalid_count > max_invalid)
			{
				return -1;
			}
		}
	}
	if (debug&&false)
	{
		std::cout << "Circle of pixels\n";
		for (int i = 0; i < 16; i++)
			std::cout << circle_of_pixels[i]<< " ";
		std::cout << "\nStatus\n";
		for (int i = 0; i < 16; i++)
			std::cout << pixel_status[i] << " ";
	}
	// Find the first change of valid status of pixels.
	// TODO: be more clever about this. at some point. maybe...
	bool found_change = false;
	int first_change_idx = 0;
	for (int i = 0; i < 16; i++)
	{
		if (pixel_status[i] != pixel_status[(i + 1) % 16])
		{
			first_change_idx = (i + 1) % 16;
			found_change = true;
			break;
		}
	}

	if (!found_change)
	{
		/**
		 * Either all are brighter or darker. It is certain that all are valid because of the max_invalid count
		 * condition. So... Sweet, we can stop now if we do not care about the score for nms.
		 */
		if (!nms)
		{
			return 1;
		}
	}

	/**
	 * Check to see if there are at least min_nr_contiguous pixels of the same status.
     * Note: pixel_status[] will be treated as a ring as opposed to an array.
	 */
	int count_array[16]; // At count_array[i] how many elements of the same type have there been.
	std::fill(count_array, count_array + 16, 0); // Set all to 0;
	int count_checked_elements = 0;
	int max_contiguous = 0; // Keep track of the longest chain.
	int idx = first_change_idx; // Start where the change is because...
	count_array[idx] = 1; // ...at this position there is only one element of this type; the one before it is of some other type.
	int previous_status = pixel_status[idx];
	int current_status;
	count_array[idx] = 1;
	int last_index;
	int next_idx;
	while (count_checked_elements < 16)
	{
		next_idx = (idx + 1) % 16;
		current_status = pixel_status[next_idx];
		if (previous_status == current_status)
		{
			count_array[next_idx] = count_array[idx] + 1;
		}
		else
		{
			if ((previous_status) != 0 && (count_array[idx] > max_contiguous))
			{
				max_contiguous = count_array[idx];
				last_index = idx;
			}
			previous_status = current_status;
			count_array[next_idx] = 1;
		}
		count_checked_elements++;
		idx = (idx + 1) % 16;
	}
	if (pixel_status[idx] != 0 && count_array[idx] > max_contiguous)
	{
		max_contiguous = count_array[idx];
		last_index = idx;
	}

	if (debug && false)
	{
		std::cout << "\nCircle of pixels\n";
		for (int i = 0; i < 16; i++)
			std::cout << circle_of_pixels[i] << " ";
		std::cout << "\nStatus\n";
		for (int i = 0; i < 16; i++)
			std::cout << std::setw(3) << pixel_status[i] << " ";
		std::cout << "\ncount_array\n";
		for (int i = 0; i < 16; i++)
			std::cout << std::setw(3) << count_array[i] << " ";
	}

	if (max_contiguous < min_nr_contiguous)
	{
		return -1;
	}
	else
	{
		if (nms)
		{
			int score = 0;
			int count = max_contiguous;
			int index = last_index;
			//std::cout << "\n score deltas:\n";
			while (count > 0)
			{
				//std::cout << ((center_pixel > circle_of_pixels[index]) ? (center_pixel - circle_of_pixels[index]) : (circle_of_pixels[index] - center_pixel)) << " ";
				score += (center_pixel > circle_of_pixels[index]) ? (center_pixel - circle_of_pixels[index]) : (circle_of_pixels[index] - center_pixel);
				index--;
				if (index < 0)
				{
					index = 15;
				}
				count--;
			}
			//std::cout << "\nscore is " << score << "\n";
			return score;
		}
		else
		{
			return 1;
		}
	}	
}