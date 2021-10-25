#pragma once

#ifndef ROB7_FAST_H
#define ROB7_FAST_H

#include <opencv2/core.hpp>
#include <vector>

class Rob7FAST
{
private:
	// The minimum number of contigous pixels in the circle
	// surrounding the circle pixel.
	int min_nr_contiguous;
	// Threshold for the center pixel's intensity.
	int t;
	// Enable non maximum suppression.
	bool nms;
	// The x, y offsets from the center pixel to all 16 pixels in
	// the circle.
	int offsets[16][2];
	// Maximum and minimum values a pixel can have. I guess 255 and 0.
	int maxIntensity, minIntensity;
	int getPixelScore(int circle_of_pixels[16], int center_pixel);

public:
	Rob7FAST(int threshold = 10, bool non_maximum_suppression = true);
	void getKeypoints(cv::Mat image, std::vector<cv::KeyPoint> keypoints);
};
#endif

