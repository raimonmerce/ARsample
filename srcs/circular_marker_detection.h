/* ****************************************** cicular_marker_detection.h *** *
 * 円形マーカーの検出クラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include "circular_marker.h"

class CircularMarkerDetection
{
 public:
  // コンストラクタ
  CircularMarkerDetection ();
  
  // デストラクタ
  ~CircularMarkerDetection ();

  // マーカー検出関数
  bool Detect(const std::vector<Ellips>&	ellipse_list,
	      cv::Mat& 				image,
	      std::vector<CircularMarker>&	marker_list);

  bool DetectRect(const std::vector<std::vector<cv::Point>>& rectangle_list,
	  cv::Mat& image,
	  std::vector<CircularMarker>& marker_list);

  // メンバ変数
  Eigen::MatrixXd A;  // 座標系の変換行列
  double radiusOuter; // 外側の円の半径
  double radiusInner; // 内側の円の半径
  bool   drawMarker;  // 検出したマーカーを描画するかどうか
  cv::Point a;
  cv::Point b;
  cv::Point c;
  cv::Point d;
};

/* *********************************** End of cicular_marker_detection.h *** */
