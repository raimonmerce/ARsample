#pragma once
/* **************************************************** cicular_marker.c *** *
 * 円形マーカークラス
 * ************************************************************************* */
#include <opencv2/opencv.hpp>
#include "rectangular_marker.h"

 /*
  * コンストラクタ
  */
RectangularMarker::RectangularMarker()
{
	R = Eigen::MatrixXd::Identity(3, 3);
	T = Eigen::VectorXd::Zero(3);
	for (int n = 0; n < 16; n++) M[n] = 0;
	M[15] = 1;
}

RectangularMarker::RectangularMarker(Eigen::MatrixXd _r1, 
	Eigen::MatrixXd _r2, 
	Eigen::MatrixXd _t, 
	std::vector<cv::Point> _rect,
	Eigen::MatrixXd _A) {

	R = Eigen::MatrixXd::Identity(3, 3);
	T = Eigen::VectorXd::Zero(3);
	for (int n = 0; n < 16; n++) M[n] = 0;
	M[15] = 1;

	r1 = _r1;
	r2 = _r2;
	t = _t;
	rect = _rect;
	A = _A;
}

/*
 * @brief A function that computes the pose of the camera
 *
 * @param[in] majorRadius    big circle radius
 * @param[in] minorRadius　　small circle radius
 * @param[in] markerPositon　Placing markers
 *
 */
void RectangularMarker::ComputeCameraParam(void) {
	// Calculate the normal vector of the pointing plane and the distance from the camera to the pointing plane

	R(0,0) = r1(0,0);
	R(0, 1) = r1(1, 0);
	R(0, 2) = r1(2, 0);

	R(1, 0) = r2(0, 0);
	R(1, 1) = r2(1, 0);
	R(1, 2) = r2(2, 0);
	Eigen::Vector3d vec1(R.row(0));
	Eigen::Vector3d vec2(R.row(1));
	R.row(2) = vec1.cross(vec2);

	//// 座標系の変換行列
	Eigen::MatrixXd R_ = Eigen::Matrix3d::Zero(3, 3);
	R_(0, 1) = 1.0;
	R_(1, 0) = 1.0;
	R_(2, 2) = -1.0;

	////// 座標系の変換
	R = R_ * R;
	T(0) = t(0, 0);
	T(1) = t(1, 0);
	T(2) = t(2, 0);
	T = R_ * T;

	M[0] = R(0, 0); M[1] = R(1, 0); M[2] = R(2, 0);
	M[4] = -R(0, 1); M[5] = -R(1, 1); M[6] = -R(2, 1);
	M[8] = R(0, 2); M[9] = R(1, 2); M[10] = R(2, 2);
	M[12] = T(0); M[13] = T(1); M[14] = T(2);
	//M[12] = 0.0; M[13] = 0.0; M[14] = -200.0;
	M[15] = 1;
	std::cout << std::to_string(M[12])  << " " << std::to_string(M[13]) << " " << std::to_string(M[14]) << " " << std::endl;
}
