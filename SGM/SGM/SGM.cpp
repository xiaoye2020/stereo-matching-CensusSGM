#include "SGM.h"

SGM::SGM()
{
}

SGM::~SGM()
{

}

//// 初始化 读入参数 和 图片
bool SGM::Initialize(string leftImagePath, string rightImage, SGMParameters param)
{
	m_LeftImage = imread(leftImagePath, cv::IMREAD_GRAYSCALE);
	m_RightImage = imread(rightImage, cv::IMREAD_GRAYSCALE);

	if (m_LeftImage.empty() || m_RightImage.empty())
	{
		cout << "读入图片失败！" << endl;
		return false;
	}

	m_ImageHeight = m_LeftImage.rows;
	m_ImageWidth = m_LeftImage.cols;
	m_Parameters = param;

	return true;
}

Mat SGM::Match()
{
	Mat censusLeft, censusRight, costVolume, disparMat, aggCostVolue;
	CensusTransform(censusLeft, censusRight);

	m_RightImage.release();

	GetCostVolume(censusLeft, censusRight, costVolume);

	censusLeft.release();
	censusRight.release();

	if (costVolume.empty())
	{
		cout << "获取代价空间失败！" << endl;
		return Mat();
	}

	CostAggregation(costVolume, m_LeftImage, aggCostVolue);

	costVolume.release();

	if (aggCostVolue.empty())
	{
		cout << "代价聚合失败！" << endl;
		return Mat();
	}

	GetDisparMat(aggCostVolue, disparMat);

	Mat result = Normalization(disparMat, m_ImageHeight, m_ImageWidth, m_Parameters.minDisparity, m_Parameters.rangeDisparity);

	aggCostVolue.release();

	return result;
}

void SGM::CensusTransform(Mat &censusLeft, Mat &censusRight)
{
	ToCensus(m_LeftImage, censusLeft, m_Parameters.censusWindowSize, m_ImageHeight, m_ImageWidth);
	ToCensus(m_RightImage, censusRight, m_Parameters.censusWindowSize, m_ImageHeight, m_ImageWidth);
}

void SGM::GetCostVolume(const Mat censusLeft, const Mat censusRight, Mat &result)
{
	int sz[] = { m_ImageHeight,m_ImageWidth ,m_Parameters.rangeDisparity };
	double  leftCensus, rightCensus;
	result.create(3, sz, CV_32FC1);

	for (int y = 0; y < m_ImageHeight; y++)
	{
		for (int x = m_Parameters.rangeDisparity - 1; x < m_ImageWidth; x++)
		{
			leftCensus = censusLeft.at<double>(y, x);
			for (int z = 0; z < m_Parameters.rangeDisparity; z++)
			{
				rightCensus = censusRight.at<double>(y, x - z);
				result.at<float>(y, x, z) = util::Hammingdst(leftCensus, rightCensus);
			}
		}
	}
}

void SGM::CostAggregation(Mat original, Mat gray, Mat &result)
{
	Mat left2Right = original.clone();
	Mat right2Left = original.clone();
	Mat top2Bottom = original.clone();
	Mat bottom2Top = original.clone();
	float * left2RightPrePtr, *right2LeftPrePtr, *top2BottomPrePtr, *bottom2TopPrePtr;
	int x2, y2;
	// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
	float cost, L1, L2, L3, LminLeft2Right = FLT_MAX, LminRight2Left = FLT_MAX, LminTop2Bottom = FLT_MAX, LminBottom2Top = FLT_MAX, L4;

	//// 从1开始 
	for (int y = 1; y < m_ImageHeight - 1; y++)
	{
		for (int x = 64; x < m_ImageWidth - 1; x++)
		{
			x2 = m_ImageWidth - x + 62;
			y2 = m_ImageHeight - y - 1;

			left2RightPrePtr = left2Right.ptr<float>(y, x - 1);
			right2LeftPrePtr = right2Left.ptr<float>(y, x2 + 1);
			top2BottomPrePtr = top2Bottom.ptr<float>(y - 1, x);
			bottom2TopPrePtr = bottom2Top.ptr<float>(y2 + 1, x);

			for (int z0 = 0; z0 < m_Parameters.rangeDisparity; z0++)
			{
				if (left2RightPrePtr[z0] < LminLeft2Right)
				{
					LminLeft2Right = left2RightPrePtr[z0];
				}

				if (right2LeftPrePtr[z0] < LminRight2Left)
				{
					LminRight2Left = right2LeftPrePtr[z0];
				}

				if (top2BottomPrePtr[z0] < LminTop2Bottom)
				{
					LminTop2Bottom = top2BottomPrePtr[z0];
				}

				if (bottom2TopPrePtr[z0] < LminBottom2Top)
				{
					LminBottom2Top = bottom2TopPrePtr[z0];
				}
			}

			for (int z = 0; z < m_Parameters.rangeDisparity; z++)
			{
#pragma region left2Right
				cost = left2Right.at<float>(y, x, z);
				L1 = left2RightPrePtr[z];
				if (z == 0)
				{
					L2 = L1 + m_Parameters.P1;
				}
				else
				{
					L2 = left2RightPrePtr[z - 1] + m_Parameters.P1;
				}

				if (z == m_Parameters.rangeDisparity - 1)
				{
					L3 = L1 + m_Parameters.P1;
				}
				else
				{
					L3 = left2RightPrePtr[z + 1] + m_Parameters.P1;
				}

				L4 = LminLeft2Right + m_Parameters.P2base / (abs(gray.at<uchar>(y, x) - gray.at<uchar>(y, x - 1)) + 1);

				left2Right.at<float>(y, x, z) = cost + min(L1, min(L2, min(L3, L4))) - LminLeft2Right;
#pragma endregion
#pragma region right2Left
				cost = right2Left.at<float>(y, x2, z);
				L1 = right2LeftPrePtr[z];
				if (z == 0)
				{
					L2 = L1 + m_Parameters.P1;
				}
				else
				{
					L2 = right2LeftPrePtr[z - 1] + m_Parameters.P1;
				}

				if (z == m_Parameters.rangeDisparity - 1)
				{
					L3 = L1 + m_Parameters.P1;
				}
				else
				{
					L3 = right2LeftPrePtr[z + 1] + m_Parameters.P1;
				}

				L4 = LminRight2Left + m_Parameters.P2base / (abs(gray.at<uchar>(y, x2) - gray.at<uchar>(y, x2 + 1)) + 1);

				right2Left.at<float>(y, x2, z) = cost + min(L1, min(L2, min(L3, L4))) - LminRight2Left;
#pragma endregion
#pragma region  top2Bottom
				cost = top2Bottom.at<float>(y, x, z);
				L1 = top2BottomPrePtr[z];
				if (z == 0)
				{
					L2 = L1 + m_Parameters.P1;
				}
				else
				{
					L2 = top2BottomPrePtr[z - 1] + m_Parameters.P1;
				}

				if (z == m_Parameters.rangeDisparity - 1)
				{
					L3 = L1 + m_Parameters.P1;
				}
				else
				{
					L3 = top2BottomPrePtr[z + 1] + m_Parameters.P1;
				}

				L4 = LminBottom2Top + m_Parameters.P2base / (abs(gray.at<uchar>(y, x) - gray.at<uchar>(y - 1, x)) + 1);

				top2Bottom.at<float>(y, x, z) = cost + min(L1, min(L2, min(L3, L4))) - LminBottom2Top;
#pragma endregion
#pragma region bottom2TopPrePtr
				cost = bottom2Top.at<float>(y2, x, z);
				L1 = bottom2TopPrePtr[z];
				if (z == 0)
				{
					L2 = L1 + m_Parameters.P1;
				}
				else
				{
					L2 = bottom2TopPrePtr[z - 1] + m_Parameters.P1;
				}

				if (z == m_Parameters.rangeDisparity - 1)
				{
					L3 = L1 + m_Parameters.P1;
				}
				else
				{
					L3 = bottom2TopPrePtr[z + 1] + m_Parameters.P1;
				}

				L4 = LminTop2Bottom + m_Parameters.P2base / (abs(gray.at<uchar>(y2, x) - gray.at<uchar>(y2 + 1, x)) + 1);

				bottom2Top.at<float>(y2, x, z) = cost + min(L1, min(L2, min(L3, L4))) - LminTop2Bottom;
#pragma endregion
			}
		}
	}

	result = left2Right + right2Left + top2Bottom + bottom2Top;
}

void SGM::GetDisparMat(Mat aggCostVolume, Mat& result)
{
	result.create(m_ImageHeight, m_ImageWidth, CV_32FC1);
	float currentVal;
	for (int y = 0; y < m_ImageHeight; y++)
	{
		for (int x = 0; x < m_ImageWidth; x++)
		{
			float min = INTMAX_MAX;
			int index = -1;
			for (int z = 0; z < m_Parameters.rangeDisparity; z++)
			{
				currentVal = aggCostVolume.at<float>(y, x, z);
				if (currentVal >= 0 && currentVal < min)
				{
					min = currentVal;
					index = z;
				}
			}

			//// 偏移0.25 0为无效值
			if (index >= 0)
			{
				result.at<float>(y, x) = index + 0.25;
			}
			else
			{
				result.at<float>(y, x) = 0;
			}
		}
	}
}





