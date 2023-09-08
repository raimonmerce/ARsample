/* **************************************************** cicular_marker.h *** *
 * 円形マーカークラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class RectangularMarker
{
public:
	// コンストラクタ
	RectangularMarker();
	RectangularMarker(Eigen::MatrixXd _r1, Eigen::MatrixXd _r2, Eigen::MatrixXd _t, std::vector<cv::Point> _rect, Eigen::MatrixXd _A);

	// カメラの位置姿勢を計算する関数
	void ComputeCameraParam(void);

	Eigen::MatrixXd r1;
	Eigen::MatrixXd r2;
	Eigen::MatrixXd t;
	std::vector<cv::Point> rect;
	Eigen::MatrixXd A;

	Eigen::MatrixXd R;
	Eigen::VectorXd T;
	float M[16];
};