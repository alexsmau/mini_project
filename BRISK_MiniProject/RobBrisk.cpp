#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

#include "RobBrisk.h"

using namespace cv;
using namespace std;

ROB_Brisk::ROB_Brisk(Mat img) 
{
	//Clone img to image
}

//Keypoint Detection
void ROB_Brisk::create_scale_space() {}
void ROB_Brisk::computeFAST() {}
void ROB_Brisk::nms_scales() {}
void ROB_Brisk::compute_subpixel_maximum() {}
void ROB_Brisk::reinterpolate() {}

//Keypoint Description
void ROB_Brisk::sampling() {}
void ROB_Brisk::pair_generation() {}
void ROB_Brisk::pair_division() {}
void ROB_Brisk::distance_computation() {}
void ROB_Brisk::descriptor() {}