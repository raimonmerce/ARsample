/* *************************************************** ellipse_fitting.c *** *
 * 楕円当てはめクラス
 * ************************************************************************* */
#include "ellipse_fitting.h"

/*
 * コンストラクタ
 */
EllipseFitting::EllipseFitting() {
  u            = Eigen::VectorXd::Zero(6);
  F0           = 1.0;
  computeError = true;
}

/*
 * デストラクタ
 */
EllipseFitting::~EllipseFitting() {
  ;
}

/*
 * データベクトルXIを計算する関数
 *
 * @param [in]  points : 点列
 * @param [in]  F0     : スケールパラメータ
 * @param [out] XI     : データベクトル
 */
static void
ComputeXI (const std::vector<cv::Point> points,
	   double			F0,
	   Eigen::MatrixXd&	  	XI) {
  int npoints = points.size();
  for (int n = 0; n < npoints; n++) {
    double x = points[n].x;
    double y = points[n].y;
    XI(0, n) = x * x;
    XI(1, n) = 2.0 * x * y;
    XI(2, n) = y * y;
    XI(3, n) = 2.0 * x * F0;
    XI(4, n) = 2.0 * y * F0;
    XI(5, n) = F0 * F0;
  }
}

/*
 * 共分散行列V0を計算する関数
 *
 * @param [in]  points : 点列
 * @param [in]  F0     : スケールパラメータ
 * @param [out] V0     : 共分散行列
 */
static void
ComputeV0(const cv::Point& 	point,
	  double		F0,
	  Eigen::MatrixXd& 	V0) {
  double x    = point.x;
  double y    = point.y;
  double xx   = x * x;
  double xy   = x * y;  
  double yy   = y * y;
  double xf0  = x * F0;
  double yf0  = y * F0;
  double f0f0 = F0 * F0;  

  V0(0, 0) = xx;
  V0(0, 1) = xy;
  V0(0, 2) = 0.0;
  V0(0, 3) = xf0;
  V0(0, 4) = 0.0;
  V0(0, 5) = 0.0;

  V0(1, 0) = xy;
  V0(1, 1) = xx + yy;
  V0(1, 2) = xy;
  V0(1, 3) = yf0;
  V0(1, 4) = xf0;
  V0(1, 5) = 0.0;

  V0(2, 0) = 0.0;
  V0(2, 1) = xy;
  V0(2, 2) = yy;
  V0(2, 3) = 0.0;
  V0(2, 4) = yf0;
  V0(2, 5) = 0.0;

  V0(3, 0) = xf0;
  V0(3, 1) = yf0;
  V0(3, 2) = 0.0;
  V0(3, 3) = f0f0;
  V0(3, 4) = 0.0;
  V0(3, 5) = 0.0;

  V0(4, 0) = 0.0;
  V0(4, 1) = xf0;
  V0(4, 2) = yf0;
  V0(4, 3) = 0.0;
  V0(4, 4) = f0f0;
  V0(4, 5) = 0.0;

  V0(5, 0) = 0.0;
  V0(5, 1) = 0.0;
  V0(5, 2) = 0.0;
  V0(5, 3) = 0.0;
  V0(5, 4) = 0.0;
  V0(5, 5) = 0.0;
}

/*
 * 楕円当てはめ関数
 *
 * @param [in] points : 楕円当てはめに使用する点列
 *
 * @return 計算したパラメータが楕円であればtrue, そうでなければfalse
 */
bool EllipseFitting::Fit(std::vector<cv::Point> points) {
  int npoints = points.size();
  Eigen::MatrixXd XI(6, npoints);
  ComputeXI (points, F0, XI);

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6, 6);
  for (int n = 0; n < npoints; n++) M += (XI.col(n) * XI.col(n).transpose());
  M /= npoints;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  u = es.eigenvectors().col(0);
  u.normalize();

  bool result = (u(0) * u(2) - u(1) * u(1) > 0) ? true : false;

  error = 0.0;
  if (computeError) {
    for (int n = 0; n < npoints; n++) {
      Eigen::MatrixXd V0(6, 6);
      ComputeV0 (points[n], F0, V0);
      double uxi = u.dot(XI.col(n));
      double uV0u = u.dot(V0 * u);
      error += (0.5 * sqrt(((uxi * uxi) / uV0u)));
    }
    error /= npoints;
  }
  return result;
}

/* ******************************************** End of ellipse_fitting.c *** */
