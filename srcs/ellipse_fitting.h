/* *************************************************** ellipse_fitting.h *** *
 * 楕円当てはめクラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class EllipseFitting
{
  // メソッド
 public:
  // コンストラクタ
  EllipseFitting();

  // デストラクタ
  ~EllipseFitting();

  // 楕円パラメータを推定する関数
  bool Fit(std::vector<cv::Point> point);

  // メンバ変数
  Eigen::VectorXd u;            // 楕円パラメータ(6次元ベクトル)
  double	  F0;           // スケールパラメータ
  double          error;        // 楕円当てはめの平均誤差
  bool            computeError; // 誤差を計算するかどうか
};

/* ******************************************** End of ellipse_fitting.h *** */
