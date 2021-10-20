/*
	BRISK(general)
	-> Feature Detection
		1.- Create scale-space Pyramid (Layers are divided into Octaves and Intra-octaves)
		2.- Compute FAST across scale-space
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
tuple <vector<KeyPoint>, vector<Mat>> FastReturn;

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

	/*for (int a = 0; a < img.size(); a++)
	{
		Mat img1 = img[a];
		imshow("img", img1);
		waitKey(0);
	}*/

	vector<vector<Mat>> octaves;
	vector<vector<Mat>> intraoctaves;

	int scales = 4;

	for (int i = 0; i < img.size(); i++)
	{
		vector<Mat> img_;
		Mat octave;

		for (int j = 0; j < scales; j++)
		{
			if (octave.empty())
			{
				pyrDown(img[i], octave, Size((img[i].cols) / 2, (img[i].rows) / 2));
				img_.push_back(octave);
			}
			else
			{
				pyrDown(octave, octave, Size((octave.cols) / 2, (octave.rows) / 2));
				img_.push_back(octave);
			}
		}
		octaves.push_back(img_);
	}

	/* show all the octaves
	for (int a = 0; a <= octaves.size(); a++)
	{
		for (int b = 0; b <= octaves.size(); b++)
		{
		Mat img3 = octaves[a][b];
				imshow("Result", img3);
				waitKey(0);
		}
	}*/
	
	for (int i = 0; i < img.size(); i++)
	{
		vector<Mat> _img;
		Mat intraoctave;

		for (int j = 0; j < scales; j++)
		{
			if (intraoctave.empty())
			{
				resize(img[i], intraoctave, Size(2 * (img[i].cols) / 3, 2 * (img[i].rows) / 3), INTER_LINEAR);
				_img.push_back(intraoctave);
			}
			else
			{
				resize(intraoctave, intraoctave, Size(2 * (intraoctave.cols) / 3, 2 * (intraoctave.rows) / 3), INTER_LINEAR);
				_img.push_back(intraoctave);
			}
		}
		intraoctaves.push_back(_img);
	}

	/* show all the intraoctaves
	for (int a = 0; a <= intraoctaves.size(); a++)
	{
		for (int b = 0; b <= intraoctaves.size(); b++)
		{
			Mat img3 = intraoctaves[a][b];
			imshow("Result", img3);
			waitKey(0);
		}
	}*/

	getscales = make_tuple(octaves, intraoctaves);
}

void computeFAST(vector<Mat> images, vector<vector<Mat>> octaves, vector<vector<Mat>> intraoctaves)
{
	vector<Mat> img;
	vector<KeyPoint> keypoints;
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
		FAST(img[i], keypoints, threshold, nms, cv::FastFeatureDetector::DetectorType::TYPE_9_16);
		//drawKeypoints(img[i], keypoints, img[i], Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//imshow("FAST", img[i]);
		//waitKey(0); 
	}
	FastReturn = make_tuple(keypoints, img);
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

	vector<KeyPoint> keypoints = get<0>(FastReturn);
	vector<Mat> orderedimg = get<1>(FastReturn);

	/*
	for (int j = 0; j < images.size(); j++)
	{
		Mat img3 = images[j];
		cout << "Displaying image" << endl;
		namedWindow("Result", WINDOW_KEEPRATIO);
		resizeWindow("Result", 1000, 1000);
		imshow("Result", img3);
		waitKey(0);
	}*/
}