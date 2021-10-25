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
	offsets[6][0] = 1;
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
}

void Rob7FAST::getKeypoints(cv::Mat image, std::vector<cv::KeyPoint> keypoints)
{
	int circle_of_pixels[16];
	int ll, cc;
	int debug_count = 0;
	try {
		if (!nms)
		{
			std::cout << "rows " << image.rows << " cols " << image.cols << "\n";
			for (int i = 3; i < (image.rows - 3); i++)
			{
				for (int j = 3; j < (image.cols - 3); j++)
				{

					for (int k = 0; k < 16; k++)
					{
						ll = i + offsets[k][0];
						cc = j + offsets[k][1];
						circle_of_pixels[k] = (int)(image.at<unsigned char>(ll, cc));
					}
					std::cout << "i " << i << " j " << j << " ll " << ll << " cc " << cc << "\n";
					if (i == 3 && j == 346)
					{
						for (int k = 0; k < 16; k++)
							std::cout << circle_of_pixels[k] << " ";
						std::cout << "\n center pix: " << ((int)(image.at<unsigned char>(i, j))) << "\n";
					}
					if (getPixelScore(circle_of_pixels, ((int)(image.at<unsigned char>(i, j)))))
					{
						debug_count++;
					}

				}
;
			}

			std::cout << " Rob7 algo has found " << debug_count << "keypoints \n";
		}
	}
	catch (...)
	{
		std::cout << "kill me\n";
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
			pixel_status[0] = 0;
			invalid_count++;
			if (invalid_count > max_invalid)
			{
				return -1;
			}
		}
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
			while (count > 0)
			{
				score += (center_pixel > circle_of_pixels[index]) ? (center_pixel - circle_of_pixels[index]) : (circle_of_pixels[index] - center_pixel);
				index--;
				if (index < 0)
				{
					index = 15;
				}
				count--;
			}

			return score;
		}
		else
		{
			return 1;
		}
	}	
}