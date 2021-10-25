#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>


using namespace cv;
using namespace std;

class ROB_Brisk
{
	
public: 
	string img_path = "..//images"; //Path to the images we want to used
	Mat img; //Variable for the image
	vector<Mat> octaves; //Vector containing the halfsampled images
	vector<Mat> intraoctaves; //Vector containing the downsampled (images by 2/3)
	vector<KeyPoint> keypoints; //Vector containing the keypoijnts obtained from FAST
	
	vector<Mat> load_img(string) {}; //Function for loading the images we want to use (MAYBE NOT NECESSARY?¿)
	
	//Feature Detection
	void create_ScaleSpace(Mat) {}; //Function for "creating" the Scale-Space pyramid
	void computeFAST() {}; //Function for computing the FAST algorithm, returning the keypoints found
	void nms_scales() {}; //Function for computing non-maximum supression between layes
	void compute_subpixel_maximum() {}; //Function for computing subpixel maximum
	void reinterpolate() {}; //Function for re-interpolating the image coordinates

	//Feature Description
	void sampling() {}; //Function for sampling points around keypoint
	void pair_generation() {}; //Function for generating pairs of points
	void distance_computation() {}; //Function for computing the overall distance of the keypoint
	void descriptor() {}; //Function for generating the Descriptor

private:
	void pair_division() {}; //Function for dividing the pairs found into Long and Short
};

int main() 
{
	ROB_Brisk *brisk1 = new ROB_Brisk();
	ROB_Brisk* brisk2 = new ROB_Brisk();
}