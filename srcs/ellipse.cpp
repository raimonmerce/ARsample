/* *********************************************************** ellipse.c *** *
 * 楕円クラス
 * ************************************************************************* */
#include "ellipse.h"
#include <algorithm>
#include <iterator>

/*
 * コンストラクタ
 */
Ellips::Ellips() {
  u  = Eigen::VectorXd::Zero(6);
  Q  = Eigen::Matrix3d::Zero(3, 3);
  cx = 0.0;
  cy = 0.0;
  majorLength = 0.0;
  minorLength = 0.0;
}

/*
 * コンストラクタ
 *
 * @param [in] _u : 楕円パラメータ
 */
Ellips::Ellips(const Eigen::VectorXd& _u) {
  u  = _u;
  Q  = Eigen::Matrix3d::Zero(3, 3);
  Q << u(0), u(1), u(3), u(1), u(2), u(4), u(3), u(4), u(5);
  cx = 0.0;
  cy = 0.0;
  majorLength = 0.0;
  minorLength = 0.0;
}

/*
 * デストラクタ
 */
Ellips::~Ellips() {
  ;
}

/*
 * 楕円パラメータを登録する関数
 *
 * @param [in] _u : 楕円パラメータ
 */
void Ellips::SetParam(const Eigen::VectorXd& _u) {
  u  = _u;
  Q << u(0), u(1), u(3), u(1), u(2), u(4), u(3), u(4), u(5);
}

/*
 * 点列を登録する関数
 *
 * @param [in] _points : 点のリスト
 */
void Ellips::SetPoints(std::vector <cv::Point> _points) {
  std::copy(_points.begin(), _points.end(), std::back_inserter(point_list));
}

/*
 * 楕円属性(楕円中心の座標, 長軸と短軸の長さ)を計算する関数
 */
void Ellips::ComputeAttributes(void) {
  double A = u(0);
  double B = u(1);
  double C = u(2);
  double D = u(3);
  double E = u(4);
  double F = u(5);

  cx = -(C * D - B * E) / (A * C - B * B);
  cy = -(A * E - B * D) / (A * C - B * B);

  double c = A * cx * cx + 2 * B * cx * cy + C * cy * cy - F;

  A /= c;
  B /= c;
  C /= c;

  double part = sqrt((A + C) * (A + C) - 4 * (A * C - B * B));
  double lambda1 = 0.5 * ((A + C) - part);
  double lambda2 = 0.5 * ((A + C) + part);

  majorLength = 1.0 / sqrt(fabs(lambda1));
  minorLength = 1.0 / sqrt(fabs(lambda2));
}

/*
 * 楕円パラメータに点(x, y)を代入した値を計算する関数
 *
 * @param [in] x, y : 点(x, y)
 *
 * @return 楕円パラメータに点(x, y)を代入した値
 */
double Ellips::EllipseValue(double x, double y) {
  double A = u(0);
  double B = u(1);
  double C = u(2);
  double D = u(3);
  double E = u(4);
  double F = u(5);

  return A * x * x + 2 * B * x * y + C * y * y + 2 * D * x + 2 * E * y + F;
}

/*
 * 点(x, y)が楕円の外部か内部かを判定する関数 
 *
 * @param [in] x, y: 点(x, y)
 *
 * @return 点(x, y)が楕円の内部であればtrue, そうでなければfalse
 */
bool Ellips::CheckInner(double x, double y) {
  static bool initialized = false;
  static double center_sgn = 0;

  if (!initialized) {
    center_sgn = EllipseValue(cx, cy);
    initialized = true;
  }
  double target_sgn = EllipseValue(x, y);

  bool retval = center_sgn * target_sgn > 0 ? true : false;

  return retval;
}

/* **************************************************** End of ellipse.c *** */
