/* **************************************************** cicular_marker.c *** *
 * 円形マーカークラス
 * ************************************************************************* */
#include <opencv2/opencv.hpp>
#include "circular_marker.h"
#include "ellipse.h"

/*
 * コンストラクタ
 */
CircularMarker::CircularMarker()
{
  R = Eigen::MatrixXd::Identity(3, 3);
  T = Eigen::VectorXd::Zero(3);
  for (int n = 0; n < 16; n++) M[n] = 0;
  M[15] = 1;
}

/*
 * コンストラクタ
 *
 * @param [in] _ellipseOuter : 大きい楕円のパラメータ
 * @param [in] _radiusOuter  : 大きい円の半径
 * @parma [in] _ellipseInner : 小さい楕円のパラメータ
 * @param [in] _radiusOuter  : 小さい円の半径
 * @param [in] _position     : マーカーの配置
 */
CircularMarker::CircularMarker(const Ellips&	_ellipseOuter,
			       double 		_radiusOuter,
			       const Ellips&	_ellipseInner,
			       double 		_radiusInner,
			       int		_position) {
  ellipseOuter = _ellipseOuter;
  ellipseInner = _ellipseInner;
  radiusOuter  = _radiusOuter;
  radiusInner  = _radiusInner;
  position     = _position;

  R = Eigen::MatrixXd::Identity(3, 3);
  T = Eigen::VectorXd::Zero(3);
  for (int n = 0; n < 16; n++) M[n] = 0;
  M[15] = 1;
}

/*
 * デストラクタ
 */
CircularMarker::~CircularMarker() {
  ;
}

/*
 * @brief  キュービックルートを計算する関数
 * 
 * @parame[in] y  x^3 = y
 *
 * @return 解
 */
static double cubicroot(double y) {
  double pre_x, x;
  double eps = 1.0e-8;
  double dx = 1.0;

  x = -1;
  while (fabs(dx) > eps) {
    pre_x = x;
    dx = (pre_x * pre_x * pre_x - y) / (3.0 * pre_x * pre_x);
    x = pre_x - dx;
  }
  return x;
}

/*
 * @brief 楕円パラメータから支持平面の法線ベクトルを計算する関数
 * 
 * @param [in/out] Q          : 楕円パラメータ
 * @param [out] v             : 法線ベクトル
 * @param [in] radius         : 円の半径
 * @parma [in] markerPosition : マーカーの配置
 *
 * @retval カメラから指示平面までの距離
 */
static double
ComputeNormal(Eigen::MatrixXd&	Q,
	      Eigen::VectorXd&	v,
	      double 		radius,
	      int 		position) {
  // 行列式が-1になるように正規化
  double param = cubicroot(-Q.determinant());
  Q /= param;
  
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
  Eigen::VectorXd u0 = es.eigenvectors().col(0);
  Eigen::VectorXd u2 = es.eigenvectors().col(2);
  Eigen::VectorXd eval = es.eigenvalues();
  
  Eigen::VectorXd v1 = sqrt((eval(2) - eval(1)) / (eval(2) - eval(0))) * u2;
  Eigen::VectorXd v2 = sqrt((eval(1) - eval(0)) / (eval(2) - eval(0))) * u0;

  v = (v1 + v2).normalized();

  if (((v(0) * v(2) > 0) && (position == 0)) ||
      ((v(0) * v(2) <= 0) && (position == 1))) {
    v = (v1 - v2).normalized();
  }
  if (v[2] > 0) v = -v;

  return sqrt(eval(1) * eval(1) * eval(1)) * radius;
}

/*
 * @brief カメラの位置姿勢を計算する関数
 *
 * @param[in] majorRadius    大きな円の半径
 * @param[in] minorRadius　　小さな円の半径
 * @param[in] markerPositon　マーカーの配置
 *
 */
void CircularMarker::ComputeCameraParam(void) {
  // 指示平面の法線ベクトルとカメラから指示平面までの距離を計算
  Eigen::VectorXd normalOuter(3), normalInner(3);
  double distOuter = ComputeNormal (ellipseOuter.Q,
				    normalOuter, radiusOuter, position);
  
  // 世界座標系のＹ軸の計算
  Eigen::Vector3d Y = normalOuter;

  // カメラから支持平面までの距離の計算
  double dist = distOuter;
  
  // 外側の円の中心を指すベクトルを計算
  Eigen::VectorXd XcOuter = ellipseOuter.Q.inverse() * Y;
  XcOuter(0) /= XcOuter(2);
  XcOuter(1) /= XcOuter(2);
  XcOuter(2) = 1.0;
  Eigen::VectorXd RcOuter = -dist * XcOuter / Y.dot(XcOuter);

  // 内側の円の中心を指すベクトルを計算
  Eigen::VectorXd XcInner = ellipseInner.Q.inverse() * Y;
  XcInner(0) /= XcInner(2);
  XcInner(1) /= XcInner(2);
  XcInner(2) = 1.0;
  Eigen::VectorXd RcInner = -dist * XcInner / Y.dot(XcInner);

  // カメラの並進ベクトルの生成
  T = RcOuter;

  // 世界座標系のＺ軸を計算（二つの円の中心を結んだ方向）
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3, 3);
  Eigen::Vector3d Z = (((I - Y * Y.transpose())
			* (RcInner - RcOuter)).normalized());

  // 世界座標系のＸ軸を計算（Ｙ軸とＺ軸の外積として計算）
  Eigen::Vector3d X = Y.cross(Z).normalized();

  // カメラの回転行列を生成
  R.col(0) = X;
  R.col(1) = Y;
  R.col(2) = Z;

  // 座標系の変換行列
  Eigen::MatrixXd R_ = Eigen::Matrix3d::Zero(3, 3);
  R_(0, 1) = 1.0;
  R_(1, 0) = 1.0;
  R_(2, 2) = -1.0;

  // 座標系の変換
  R = R_ * R;
  T = R_ * T;
	   
  // モデルビュー行列の生成
  M[0] = R(0, 0); M[1] = R(1, 0); M[2]  = R(2, 0);
  M[4] = R(0, 1); M[5] = R(1, 1); M[6]  = R(2, 1);
  M[8] = R(0, 2); M[9] = R(1, 2); M[10] = R(2, 2);
  M[12] = T(0); M[13] = T(1); M[14] = T(2);
}

/* ********************************************* End of cicular_marker.c *** */
