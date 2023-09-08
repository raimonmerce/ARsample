/* **************************************************** cicular_marker.h *** *
 * 円形マーカークラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "ellipse.h"

class CircularMarker
{
 public:
  // コンストラクタ
  CircularMarker();
  CircularMarker(const Ellips& _ellipseOuter, double _radiusOuter,
		 const Ellips& _ellipseInner, double _radiusInner,
		 int _position);

  // デストラクタ
  ~CircularMarker();

  // カメラの位置姿勢を計算する関数
  void ComputeCameraParam(void);

  // メンバ変数
  Ellips ellipseOuter;  // 大きな楕円のパラメータ
  Ellips ellipseInner;  // 小さな楕円のパラメータ
  double  radiusOuter;   // 大きな円の半径
  double  radiusInner;   // 小さな円の半径
  int     position;      // マーカーの配置（水平:0, 垂直:1）
  Eigen::MatrixXd R;     // カメラの回転行列
  Eigen::VectorXd T;     // カメラの併進ベクトル
  float           M[16]; // モデルビュー行列（OpenGLで使用）
};

/* ********************************************* End of cicular_marker.h *** */
