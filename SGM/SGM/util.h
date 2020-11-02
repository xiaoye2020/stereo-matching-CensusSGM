#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
namespace util {
	void ToCensus(Mat matSor, Mat &modified_image, int windowSize, int image_height, int image_width);

	Mat  Normalization(Mat dispMat, int image_height, int image_width, int mindispar, int range);

	int  Hammingdst(long long PL, long long PR);
}