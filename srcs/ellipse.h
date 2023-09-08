/* *********************************************************** ellipse.h *** *
 * 楕円クラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class Ellips
{
 public:
  // コンストラクタ
  Ellips();
  Ellips(const Eigen::VectorXd& _u);

  // デストラクタ
  ~Ellips();

  // 楕円パラメータを登録する関数
  void   SetParam		(const Eigen::VectorXd& _u);
  // 点列を登録する関数
  void   SetPoints		(std::vector<cv::Point> _points);
  // 楕円属性を計算する関数
  void   ComputeAttributes	(void);
  // 楕円パラメータに点(x, y)を代入した値を計算する関数
  double EllipseValue 		(double x, double y);
  // 点(x, y)が楕円の内部にあるかどうか調べる関数
  bool   CheckInner		(double x, double y);

  // メンバ変数
  Eigen::VectorXd u; // 楕円パラメータ(6次元ベクトル)
  Eigen::MatrixXd Q; // 楕円パラメータ(3x3行列)
  double cx;         // 楕円中心の座標(cx, cy)
  double cy;
  double majorLength;// 長軸の長さ
  double minorLength;// 短軸の長さ
  std::vector<cv::Point> point_list; // 点列データ
};

/* *************************************************** End of ellipse.h *** */
