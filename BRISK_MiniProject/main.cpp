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

using namespace cv;
using namespace std;

tuple <vector<vector<Mat>>, vector<vector<Mat>>> getscales;
tuple <vector<Mat>> FastReturn;

vector<vector<KeyPoint>> keypoints;

vector<Mat> load_img(string img_path)
{
	vector<Mat> images;
	vector<string> path;

	glob(img_path, path);

	size_t number_of_img = path.size();

	for (int j = 0; j < number_of_img; j++)
	{
		Mat img = imread(path[j], IMREAD_GRAYSCALE);
		images.push_back(img);
	}

	return images;
}

void create_ScaleSpace(vector<Mat> images)
{
	vector<Mat> img = images;

	vector<vector<Mat>> octaves;
	vector<vector<Mat>> intraoctaves;

	int scales = 4;

	for (int i = 0; i < img.size(); i++)
	{
		vector<Mat> img_;
		Mat octave;
		vector<Mat> _img;
		Mat intraoctave;

		for (int j = 0; j < scales; j++)
		{
			if (octave.empty())
			{
				pyrDown(img[i], octave, Size((img[i].cols) / 2, (img[i].rows) / 2));
				img_.push_back(octave);
				resize(octave, intraoctave, Size(2 * (octave.cols) / 3, 2 * (octave.rows) / 3), INTER_LINEAR);
				_img.push_back(intraoctave);
			}
			else
			{
				pyrDown(octave, octave, Size((octave.cols) / 2, (octave.rows) / 2));
				img_.push_back(octave);
				resize(octave, intraoctave, Size(2 * (octave.cols) / 3, 2 * (octave.rows) / 3), INTER_LINEAR);
				_img.push_back(intraoctave);
			}
		}
		octaves.push_back(img_);
		intraoctaves.push_back(_img);
	}

	getscales = make_tuple(octaves, intraoctaves);
}

void computeFAST(vector<Mat> images, vector<vector<Mat>> octaves, vector<vector<Mat>> intraoctaves)
{
	vector<Mat> img;
	vector<KeyPoint> keypoint;
	vector<vector<vector<KeyPoint>>> outputkeypoints;
	int threshold = 50;
	bool nms = true;

	for (int i = 0; i < images.size(); i++)
	{
		img.push_back(images[i]);

		for (int j = 0; j < 4; j++)
		{
			img.push_back(octaves[i][j]);
			img.push_back(intraoctaves[i][j]);
		}
	}

	for (int i = 0; i < img.size(); i++)
	{
		FAST(img[i], keypoint, threshold, nms, cv::FastFeatureDetector::DetectorType::TYPE_9_16);
		keypoints.push_back(keypoint);

		//drawKeypoints(img[i], keypoints, img[i], Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//imshow("FAST", img[i]);
		//waitKey(0); 
	}
	FastReturn = make_tuple(img);
}

void nms_scales(vector<Mat> img)
{
	cout << img.size() << endl;
	for (int i = 0; i < img.size(); i++)
	{
		vector<KeyPoint> key = keypoints[i];
		vector<KeyPoint> downscaled;
		vector<KeyPoint> upscaled;

		int k = i + 1;
		if (k < img.size())
		{
			downscaled = keypoints[k];
		}
		int m = i - 1;
		if (m >= 0)
		{
			upscaled = keypoints[m];
		}

		/*
			PROBLEMS:
			- We need to have access to the scores of all the keypoints found by FAST (Alex?¿)
			- Does there exist a function that computes NMS between scales?
			- We need an easy way of NMS2.0, compare the keypoints on adyacent scales and remove the smaller ones
			- - std::sort()
		*/
		for (int j = 0; j < keypoints[i].size(); j++)
		{
			if (i < 9)
			{
				if (i == 0) //Compare with one above
				{
					for (int a = 0; a < downscaled.size(); a++)
					{
						float topleft[2] = { round(keypoints[i][j].pt.x / 2) - 1 , round(keypoints[i][j].pt.y / 2) + 1 };
						for (int x = 0; x < 9; x++)
						{
							for (int b = 0; b < 3; b++)
							{
								topleft[0] == downscaled[a].pt.x;
								topleft[1] == downscaled[a].pt.y;
								topleft[1] - 1 == downscaled[a].pt.y;
								topleft[1] - 2 == downscaled[a].pt.y;

								if (topleft[0] == downscaled[a].pt.x && topleft[1] - b == downscaled[a].pt.y)
								{
									//round(keypoints[i][j].pt.x / 2) == downscaled[a].pt.x;
									//round(keypoints[i][j].pt.y / 2) == downscaled[a].pt.y;
								}
								topleft[0]++;
							}
						}
					}
				}
				else if (i < 8) //Compare above and below
				{

				}
				else
				{

				}
			}
			else
			{
				//keypoints.size() >= 9;
			}
		}
	}
}

void cheatBrisk(vector<Mat> images)
{
	std::vector<cv::KeyPoint> keypointsA, keypointsB;
	cv::Mat descriptorsA, descriptorsB;

	int Threshl = 100;
	int Octaves = 4; //(pyramid layer) from which the keypoint has been extracted
	float PatternScales = 1.0f;
	Ptr<Feature2D> detector = BRISK::create(Threshl, Octaves, PatternScales);

	detector->detect(images[4], keypointsA);
	detector->detect(images[5], keypointsB);
	detector->compute(images[4], keypointsA, descriptorsA);
	detector->compute(images[5], keypointsB, descriptorsB);

	std::vector<cv::DMatch> matches;
	cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
	matcher.match(descriptorsA, descriptorsB, matches);

	cv::Mat all_matches;
	cv::drawMatches(images[4], keypointsA, images[5], keypointsB, matches, all_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::imshow("BRISK All Matches", all_matches);
	cv::waitKey(0);
}

int main()
{
	string load_img_folder = "C://Users//alber//Desktop//MiniProject//mini_project//BRISK_MiniProject//images";
	string save_img = load_img_folder + "result";

	vector<Mat> images = load_img(load_img_folder);

	create_ScaleSpace(images);

	vector<vector<Mat>> octaves = get<0>(getscales);
	vector<vector<Mat>> intraoctaves = get<1>(getscales);

	computeFAST(images, octaves, intraoctaves);

	vector<Mat> orderedimg = get<0>(FastReturn);
	//nms_scales(orderedimg);
	/*
	for (int i = 0; i < keypoints.size(); i++)
	{
		cout << keypoints[i].size() << endl;
	}

	for (int j = 0; j < orderedimg.size(); j++)
	{
		Mat img3 = orderedimg[j];
		imshow("Result", img3);
		waitKey(0);
	}
	*/

	//cheatBrisk(orderedimg);
}