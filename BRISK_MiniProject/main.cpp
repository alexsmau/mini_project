/*
	BRISK(general)
	-> Feature Detection
		1.- Create scale-space Pyramid (Layers are divided into Octaves and Intra-octaves)
		2.- Compute FAST across scale-space (scores)
		3.- Pixel level non-maximal suppression
		4.- Compute sub-pixel maximum across patch
		5.- Re-interpolate image coordinates from scale-space

	-> Feature Description
		1.- Sampling around keypoint
		2.- Generate pairs of points
		3.- Divide into Short- and Long-distance pairs
		4.- Use Long-distance pairs to compute overall distance of the keypoint
		5.- Generate the Descriptor
*/

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "RobBrisk.h"
#include "Rob7BriskDescriptor.h"

using namespace cv;
using namespace std;

int main()
{	
#if 0
	char asmau_img_path[] = "S:\\AAU\\Year1\\mini_project\\BRISK_MiniProject\\fast_cpp\\FASTcpp\\FASTcpp\\IMG_20201231_194210.jpg";
	char alberto_path[] = "..//BRISK_MiniProject//images//Dog.jpg";

	Mat image = imread(alberto_path, IMREAD_GRAYSCALE);
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	ROB_Brisk brisk1 = ROB_Brisk(image);
	brisk1.calculate_descriptors();
#else
	char asmau_img_path[] = "S:\\AAU\\Year1\\mini_project\\BRISK_MiniProject\\fast_cpp\\FASTcpp\\FASTcpp\\IMG_20201231_194210.jpg";
	char book_small[] = "S:\\AAU\\Year1\\mini_project\\BRISK_MiniProject\\images\\book_small.jpg";
	char cropped_book[] = "S:\\AAU\\Year1\\mini_project\\BRISK_MiniProject\\images\\cropped_book.jpg";

	Mat image1 = imread(book_small, IMREAD_GRAYSCALE);
	if (image1.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	Mat image2 = imread(cropped_book, IMREAD_GRAYSCALE);
	if (image2.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	ROB_Brisk brisk1 = ROB_Brisk(image1);
	brisk1.calculate_descriptors();

	vector<KeyPoint> adjusted_kp1;
	for (Rob7BriskDescriptor descr1 : brisk1.descriptors)
	{
		adjusted_kp1.push_back(KeyPoint((descr1.keypoint.pt.x * pow(2, descr1.keypoint.size)), (descr1.keypoint.pt.y * pow(2, descr1.keypoint.size)), 1, 100, 0, -1));
	}

	ROB_Brisk brisk2 = ROB_Brisk(image2);
	brisk2.calculate_descriptors();

	vector<KeyPoint> adjusted_kp2;
	for (Rob7BriskDescriptor descr1 : brisk2.descriptors)
	{
		adjusted_kp2.push_back(KeyPoint((descr1.keypoint.pt.x * pow(2, descr1.keypoint.size)), (descr1.keypoint.pt.y * pow(2, descr1.keypoint.size)), 1, 100, 0, -1));
	}

	vector<KeyPoint> match_img1;
	match_img1.clear();
	vector<KeyPoint> match_img2;
	match_img2.clear();

	vector<DMatch> matches;
	matches.clear();

	int min_score, score;
	KeyPoint matched_kp;
	int idx = 0;
	cout << "Brisk 1 has " << brisk1.descriptors.size() << " elements\n";
	cout << "Brisk 2 has " << brisk2.descriptors.size() << " elements\n";
	for (Rob7BriskDescriptor descr1 : brisk1.descriptors)
	{
		min_score = 512;
		for (Rob7BriskDescriptor descr2 : brisk2.descriptors)
		{
			score = descr1.compareTo(descr2);
			if (score < min_score)
			{
				min_score = score;
				matched_kp = descr2.keypoint;
			}
		}
		if (min_score < 180)
		{
			match_img1.push_back(KeyPoint((descr1.keypoint.pt.x * pow(2, descr1.keypoint.size)), (descr1.keypoint.pt.y * pow(2, descr1.keypoint.size)), 1, 100, 0, -1));
			match_img2.push_back(KeyPoint((matched_kp.pt.x * pow(2, matched_kp.size)), (matched_kp.pt.y * pow(2, matched_kp.size)), 1, 100, 0, -1));
			matches.push_back(DMatch(idx, idx, min_score));
			idx++;
		}
	}
	cout << "there are " << idx << " matches\n";
	Mat result;
	drawMatches(image1, match_img1, image2, match_img2, matches, result);
	
	imshow("Result", result);
	waitKey(0);
#endif


}