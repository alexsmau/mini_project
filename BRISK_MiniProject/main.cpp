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
		//bool done = false;

		for (int j = 0; j < scales; j++)
		{
			Mat octave;
			octave = Mat::zeros(0,0, CV_8UC1);

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
			//done = true;
		}
		octaves.push_back(img_);
		//if (done == false) break;
	}

	cout << "Number of octaves" << octaves.size() << endl;

	for (int a = 0; a < octaves.size(); a++)
	{
		for (int b = 0; b < octaves.size(); b++)
		{
		Mat img3 = octaves[a][b];
				imshow("Result", img3);
				waitKey(0);
		}

	}
	/*
	for (int i = 0; i < octaves.size(); i++)
	{
		vector<Mat> img_;
		bool done = false;

		for (int j = 0; j < scales; j++)
		{
			Mat intraoctave;
			pyrDown(img[i], intraoctave, Size(2*(octaves[i].cols) / 3, 2*(octaves[i].rows) / 3));
			img_.push_back(intraoctave);
			done = true;
		}

		if (done == false) break;
		intraoctaves.push_back(img_);
	}*/

}

int main()
{
	string load_img_folder = "C://Users//alber//Desktop//MiniProject//mini_project//BRISK_MiniProject//images";
	string save_img = load_img_folder + "result";

	vector<Mat> images = load_img(load_img_folder);
	create_ScaleSpace(images);

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