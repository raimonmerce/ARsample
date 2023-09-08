/* ************************************************* ellipse_detection.h *** *
 * 楕円検出クラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <math.h>
#include "ellipse_fitting.h"
#include "ellipse.h"
#include "circular_marker.h"
#include "rectangular_marker.h"

class EllipseDetection
{
  // メソッド
 public:
  // コンストラクタ
  EllipseDetection();

  // デストラクタ
  ~EllipseDetection();

  // 楕円検出
  bool Detect (cv::Mat& input, std::vector<Ellips>& ellipse_list);

  bool DetectRect(cv::Mat& input, std::vector<std::vector<cv::Point>>& rectangle_list,
	  std::vector<RectangularMarker>& marker_list, Eigen::MatrixXd _A);

  // メンバ変数
  int    minLength;          // エッジ点列の最小点数  
  double cannyParam[2];      // Cannyオペレータのパラメータ
  int    gaussianKernelSize; // ガウシアンフィルタのパラメータ
  double gaussianSigma;
  double axisRatio;          // 楕円を判定するためのパラメータ
  double axisLength;
  double errorThreshold;
  EllipseFitting ellipse_fitting;    // 楕円当てはめクラス
  std::vector<Ellips> ellipse_list; // 検出した楕円のリスト
  bool drawEllipseCenter;            // 描画フラグ
  cv::Point p1;
  cv::Point p2;
  cv::Point p3;
  cv::Point p4;
  double Pi;
  cv::Point a;
  cv::Point b;
  cv::Point c;
  cv::Point d;
  std::vector<cv::Point> corners;
  Eigen::MatrixXd A;
  // デフォルトパラメータ
  const int    DEFAULT_MIN_LENGTH           = 50;
  const int    DEFAULT_GAUSSIAN_KERNEL_SIZE = 9;
  const double DEFAULT_GAUSSIAN_SIGMA       = 1.0;
  const int    DEFAULT_CANNY_PARAM1         = 50;
  const int    DEFAULT_CANNY_PARAM2         = 200;
  const double DEFAULT_AXIS_RATIO           = 0.3;
  const double DEFAULT_AXIS_LENGTH          = 20.0;
  const double DEFAULT_ERROR_THRESHOLD      = 1.4;
};
