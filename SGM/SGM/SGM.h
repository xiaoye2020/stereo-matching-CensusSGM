#include  <string>
#include "util.h"
using namespace std;
using namespace util;

class SGM
{
public:
	SGM();
	~SGM();

	struct  SGMParameters
	{
		int minDisparity;

		int rangeDisparity;

		int censusWindowSize;

		int P1;

		int P2base;
	};

	bool Initialize(string leftImagePath, string rightImage, SGMParameters param);

	Mat Match();

private:

	SGMParameters m_Parameters;

	int m_ImageWidth;

	int m_ImageHeight;

	Mat m_LeftImage;

	Mat m_RightImage;

	void CensusTransform(Mat &censusLeft, Mat &censusRight);

	void GetCostVolume(const Mat censusLeft, const Mat censusRight, Mat &costVolume);

	void CostAggregation(Mat original, Mat gray, Mat &result);

	void GetDisparMat(Mat aggCostVolume, Mat &DisparMat);
};
