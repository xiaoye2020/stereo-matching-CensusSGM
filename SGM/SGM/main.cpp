#include <iostream>
#include "SGM.h"
//// 入口
int main() {
	//// 路径设置
	string sLeftImagePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\im2.png";
	string sRightImagePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\im6.png";
	string sSavePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\result2.png";

	//// 参数设置
	SGM::SGMParameters param;
	param.censusWindowSize = 5;
	param.minDisparity = 0;
	param.rangeDisparity = 64;
	param.P1 = 10;
	param.P2base = 150;

	//// 创建半全局匹配类
	SGM sgm;

	//// 初始化
	if (!sgm.Initialize(sLeftImagePath, sRightImagePath, param))
	{
		system("pause");
		return 0;
	}

	//// 获得视差图
	Mat result = sgm.Match();

	if (result.empty())
	{
		cout << "获取视差图失败！" << endl;
		cv::waitKey(0);
		return 0;
	}

	//// 展示保存结果
	imshow("result2", result);
	imwrite(sSavePath, result);

	cv::waitKey(0);
	return 0;
}