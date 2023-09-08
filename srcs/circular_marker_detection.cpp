/* ****************************************** cicular_marker_detection.c *** *
 * 円形マーカーの検出クラス
 * ************************************************************************* */
#include "circular_marker_detection.h"

/*
 * コンストラクタ
 */
CircularMarkerDetection::CircularMarkerDetection() {
  radiusOuter = 27.5;
  radiusInner = 15.0;
  drawMarker  = false;
  A = Eigen::MatrixXd::Identity(3, 3);
  a = cv::Point(0,0);
  b = cv::Point(720, 0);
  c = cv::Point(720, 720);
  d = cv::Point(0, 720);
}

/*
 * デストラクタ
 */
CircularMarkerDetection::~CircularMarkerDetection() {
  ;
}

/*
 * 楕円パラメータの座標系変換
 *
 * @param [in] input : 楕円パラメータ
 * @param [in] A     : 座標系の変換行列
 *
 * @return 座標系を変換した楕円
 */
static Ellips
ConvertCoordinate (const Ellips& 		input,
		   const Eigen::MatrixXd&	A) {
  Eigen::MatrixXd Q(3, 3);
  Q = A.transpose() * input.Q * A;
  Eigen::VectorXd u(6);
  u << Q(0, 0), Q(0, 1), Q(1, 1), Q(0, 2), Q(1, 2), Q(2, 2);
  u.normalize();

  Ellips ell = Ellips(u);
  ell.cx = input.cx;
  ell.cy = input.cy;

  return ell;
}

/*
 * 円形マーカーを検出する関数
 *
 * @param [in] ellipse_list : 楕円のリスト
 * @param [in] image        : 画像
 * @param [out] marker_list : マーカーのリスト
 *
 * @return マーカーが検出されればtrue, そうでなければfalse
 */
bool CircularMarkerDetection::Detect (const std::vector<Ellips>& ellipse_list,
				      cv::Mat&              image,
				      std::vector<CircularMarker>& marker_list)
{
  // リストのクリア
  marker_list.clear();

  // index indicating whether it is a processed ellipse
  Eigen::VectorXi use_index = Eigen::VectorXi::Ones(ellipse_list.size());

  for (int n = 0; n < (int) ellipse_list.size() - 1; n++) {
    if (use_index[n] == 0) continue;

    Ellips target = ellipse_list[n];
    double target_sgn = target.EllipseValue (target.cx, target.cy);
    for (int m = n + 1; m < (int) ellipse_list.size(); m++) {
      if (use_index[m] == 0) continue;

      Ellips reff = ellipse_list[m];
      double reff_sgn = target.EllipseValue (reff.cx, reff.cy);
      if (target_sgn * reff_sgn  > 0) {
	Ellips ellOuter = ConvertCoordinate (target, A);
	Ellips ellInner = ConvertCoordinate (reff, A);	
	CircularMarker marker(ellOuter, radiusOuter, ellInner, radiusInner, 0);
	marker_list.push_back(marker);
	use_index[n] = 0;
	use_index[m] = 0;
	break;
      }
    }
  }
  if (marker_list.size() == 0) return false;

  if (drawMarker) {
    for (int n = 0; n < (int) marker_list.size(); n++) {
      Ellips ell = marker_list[n].ellipseOuter;
      cv::Point p;
      p.x = ell.cx;
      p.y = ell.cy;
      cv::circle(image, p, 5, cv::Scalar(0, 0, 255), 1, 1);
    }
  }
  return true;
}

bool CircularMarkerDetection::DetectRect(const std::vector<std::vector<cv::Point>>& rectangle_list,
    cv::Mat& image,
    std::vector<CircularMarker>& marker_list)
{
    return true;
}

