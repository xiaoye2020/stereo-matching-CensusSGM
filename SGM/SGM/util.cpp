#include "util.h"

void util::ToCensus(Mat matSor, Mat &modified_image, int windowSize, int image_height, int image_width)
{
	modified_image = Mat::zeros(image_height, image_width, CV_64FC1);
	int offset = windowSize / 2;
	Mat MakeBorder;
	cv::copyMakeBorder(matSor, MakeBorder, offset, offset, offset, offset, BORDER_CONSTANT, 0);

	for (int j = 0; j < image_width - windowSize; j++)
	{
		for (int i = 0; i < image_height - windowSize; i++)
		{
			unsigned long census = 0;
			uchar current_pixel = MakeBorder.at<uchar>(i + offset, j + offset);
			Rect roi(j, i, windowSize, windowSize); //方形窗口
			Mat window(MakeBorder, roi);

			for (int a = 0; a < windowSize; a++)
			{
				for (int b = 0; b < windowSize; b++)
				{
					if (!(a == offset && b == offset))//中心像素不做判断
					{
						census = census << 1;//左移1位
					}
					uchar temp_value = window.at<uchar>(a, b);
					if (temp_value <= current_pixel) //当前像素小于中心像素 01
					{
						census += 1;
					}
				}
			}

			modified_image.at<double>(i + offset, j + offset) = census;
		}
	}
}

Mat util::Normalization(Mat dispMat, int image_height, int image_width, int mindispar, int range)
{
	Mat result(image_height, image_width, CV_8UC1);
	float currentVal, fResult;
	int iTemp;
	for (int y = 0; y < image_height; y++)
	{
		for (int x = 0; x < image_width; x++)
		{
			currentVal = dispMat.at<float>(y, x);
			if (currentVal != 0)
			{
				fResult = ((currentVal - mindispar) / range) * 255;
				iTemp = (int)fResult;
				result.at<uchar>(y, x) = (unsigned char)iTemp;
			}
			else
			{
				result.at<uchar>(y, x) = (unsigned char)0;
			}
		}
	}

	return result;
}

int util::Hammingdst(long long PL, long long PR)
{
	int number = 0;
	long long v;
	v = PL ^ PR;

	while (v)
	{
		v &= (v - 1);
		number++;
	}

	return number;
}
