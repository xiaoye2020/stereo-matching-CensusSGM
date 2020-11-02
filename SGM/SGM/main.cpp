#include <iostream>
#include "SGM.h"
//// ���
int main() {
	//// ·������
	string sLeftImagePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\im2.png";
	string sRightImagePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\im6.png";
	string sSavePath = "E:\\GitHub\\stereo-matching-CensusSGM\\Data\\cone\\result2.png";

	//// ��������
	SGM::SGMParameters param;
	param.censusWindowSize = 5;
	param.minDisparity = 0;
	param.rangeDisparity = 64;
	param.P1 = 10;
	param.P2base = 150;

	//// ������ȫ��ƥ����
	SGM sgm;

	//// ��ʼ��
	if (!sgm.Initialize(sLeftImagePath, sRightImagePath, param))
	{
		system("pause");
		return 0;
	}

	//// ����Ӳ�ͼ
	Mat result = sgm.Match();

	if (result.empty())
	{
		cout << "��ȡ�Ӳ�ͼʧ�ܣ�" << endl;
		cv::waitKey(0);
		return 0;
	}

	//// չʾ������
	imshow("result2", result);
	imwrite(sSavePath, result);

	cv::waitKey(0);
	return 0;
}