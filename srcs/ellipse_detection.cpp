/* ************************************************* ellipse_detection.c *** *
 * 楕円検出クラス
 * ************************************************************************* */

#include <Eigen/Dense>
#include "ellipse_detection.h"
#include "ellipse_fitting.h"
#include "ellipse.h"
#include <algorithm>
#include <iterator>
#include <opencv2/features2d/features2d.hpp>

 #include <opencv2/core/eigen.hpp>

 /*
  * コンストラク
  */
EllipseDetection::EllipseDetection() {
    minLength = DEFAULT_MIN_LENGTH;
    gaussianKernelSize = DEFAULT_GAUSSIAN_KERNEL_SIZE;
    gaussianSigma = DEFAULT_GAUSSIAN_SIGMA;
    cannyParam[0] = DEFAULT_CANNY_PARAM1;
    cannyParam[1] = DEFAULT_CANNY_PARAM2;
    axisRatio = DEFAULT_AXIS_RATIO;
    axisLength = DEFAULT_AXIS_LENGTH;
    errorThreshold = DEFAULT_ERROR_THRESHOLD;
    drawEllipseCenter = false;
    ellipse_fitting.computeError = true;
    Pi = acos(-1);
    int rectSize = 20;
    p1.x = 0;
    p1.y = 0;
    p2.x = rectSize;
    p2.y = 0;
    p3.x = rectSize;
    p3.y = rectSize;
    p4.x = 0;
    p4.y = rectSize;

    a = cv::Point(0, 0);
    b = cv::Point(40, 0);
    c = cv::Point(40, 40);
    d = cv::Point(0, 40);

    corners.push_back(a);
    corners.push_back(b);
    corners.push_back(c);
    corners.push_back(d);
}

/*
 * デストラクタ
 */
EllipseDetection::~EllipseDetection() {
    ;
}

static bool compareEllipseSize(const Ellips& e1, const Ellips& e2)
{
    return (e1.majorLength > e2.majorLength);
}

int lengthSquare(cv::Point X, cv::Point Y)
{
    int xDiff = X.x - Y.x;
    int yDiff = X.y - Y.y;
    return xDiff * xDiff + yDiff * yDiff;
}


static double checkAngles(cv::Point A, cv::Point B, cv::Point C, double Pi)
{
    int a2 = lengthSquare(B, C);
    int b2 = lengthSquare(A, C);
    int c2 = lengthSquare(A, B);

    // length of sides be a, b, c
    float a = sqrt(a2);
    float b = sqrt(b2);
    float c = sqrt(c2);
    int minSize = 30;
    if (a < minSize || b < minSize || c < minSize) return false;

    double res = acos((b2 + c2 - a2) / (2 * b * c));

    // Answer is in the range of [-pi...pi]

    return (res * 180 / Pi) > 15.0;
}

bool EllipseDetection::DetectRect(cv::Mat& image,
    std::vector<std::vector<cv::Point>>& rectangle_list,
    std::vector<RectangularMarker>& marker_list,
    Eigen::MatrixXd _A) {
    // Conversion to grayscale image
    A = _A;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

    // Smoothing process
    cv::Mat edge;
    cv::GaussianBlur(gray, edge,
        cv::Size(gaussianKernelSize, gaussianKernelSize),
        gaussianSigma);

    // Edge detection
    cv::Canny(edge, edge, cannyParam[0], cannyParam[1]);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> polyCurves;

    // Get poly curves for every contour
    for (int i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point> current;
        approxPolyDP(contours[i], current, 0.05 * cv::arcLength(contours[i], true), true);
        //approxPolyDP(contours[i], current, 2.0, true);

        if (current.size() > 0) {
            polyCurves.push_back(current);
        }
    }

    //Check rectangles
    std::vector<std::vector<cv::Point>> rectangles;
    if (polyCurves.size() > 0) {
        for (int i = 0; i < polyCurves.size(); ++i) {
            if (polyCurves[i].size() == 4) {
                cv::Point p1 = polyCurves[i][0];
                cv::Point p2 = polyCurves[i][1];
                cv::Point p3 = polyCurves[i][2];
                cv::Point p4 = polyCurves[i][3];
                if (checkAngles(p1, p4, p2, Pi) && checkAngles(p2, p1, p3, Pi) && checkAngles(p3, p2, p4, Pi) && checkAngles(p4, p3, p1, Pi))
                    rectangles.push_back(polyCurves[i]);
            }
        }
    }
    else return false;
    
    if (drawEllipseCenter) {
        for (int n = 0; n < rectangles.size(); n++) {
            cv::Scalar color = cv::Scalar(0, 255, 0);
            cv::drawContours(image, rectangles, n, color, 3, 8);
        }
    }
    
    marker_list.clear();

    for (int i = 0; i < (int)rectangles.size(); i++) {
        
        std::vector<cv::Point> tmp = rectangles[i];
        Eigen::Matrix3d H_eigen;

        cv::Mat H = cv::findHomography(corners, tmp, 0);
        cv::cv2eigen(H, H_eigen);

        Eigen::MatrixXd mat1(8, 9);
        mat1 << Eigen::MatrixXd::Zero(8, 9);
        
        for (int j = 0; j < 4; j++) {
            mat1(i, 0) = mat1(i + 1, 3) = (double)tmp[j].x;
            mat1(i, 1) = mat1(i + 1, 4) = (double)tmp[j].y;
            mat1(i, 2) = mat1(i + 1, 5) = 1.0;
            mat1(i, 6) = -(double)tmp[j].x * (double)corners[j].x;
            mat1(i, 7) = -(double)tmp[j].y * (double)corners[j].x;
            mat1(i, 8) = -(double)corners[j].x;
            mat1(i + 1, 6) = -(double)tmp[j].x * (double)corners[j].y;
            mat1(i + 1, 7) = -(double)tmp[j].y * (double)corners[j].y;
            mat1(i + 1, 8) = -(double)corners[j].y;
        }

        //Eigen::MatrixXf m = Eigen::MatrixXf::Random(8, 9);
        //Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU);
        Eigen:: JacobiSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(mat1);
        //Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat1);
        
        Eigen::VectorXd V(8);
        Eigen::VectorXd zero = Eigen::VectorXd::Zero(8);

        Eigen::VectorXd h = svd.solve(zero);
        H_eigen = h.reshaped(3, 3);
        
        
        Eigen::MatrixXd h1 = H_eigen.col(0);
        Eigen::MatrixXd h2 = H_eigen.col(1);
        Eigen::MatrixXd h3 = H_eigen.col(2);

        Eigen::MatrixXd tmpL = h1.transpose() * A.transpose().inverse() * A.inverse() * h1;
        double lambda = sqrt(tmpL.coeff(0, 0));
        Eigen::MatrixXd r1 = (1.0/lambda)* A.inverse()*h1;
        Eigen::MatrixXd r2 = (1.0 / lambda) * A.inverse() * h2;
        Eigen::MatrixXd t = (1.0 / lambda) * A.inverse
        () * h3;
        
        RectangularMarker marker(r1, r2, t, tmp, A);
        marker_list.push_back(marker);
    }
    
    return true;
}
/*
 * Ellipse detection function
 *
 * @param [in] image         : 入力画像
 * @param [out] ellipse_list : 検出した楕円のリスト
 *
 * @returntrue if 2 or more ellipses are detected, false otherwise
 */
bool EllipseDetection::Detect(cv::Mat& image,
    std::vector<Ellips>& ellipse_list) {
    // Conversion to grayscale image
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

    // Smoothing process
    cv::Mat edge;
    cv::GaussianBlur(gray, edge,
        cv::Size(gaussianKernelSize, gaussianKernelSize),
        gaussianSigma);

    // Edge detection
    cv::Canny(edge, edge, cannyParam[0], cannyParam[1]);

    // outline extraction
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // add to the list the sequence of points that fit the ellipse that satisfies the condition
    std::vector<Ellips> candidate_list;
    for (int n = 0; n < contours.size(); n++) {

        //Do not process if the number of point sequences is less than the threshold(minLength)
        if (contours[n].size() < minLength) continue;
        // ellipse fitting
        bool result = ellipse_fitting.Fit(contours[n]);

        // add to the list the sequence of points that fit the ellipse that satisfies the condition
        if (result && ellipse_fitting.error < errorThreshold) {
            Ellips ell(ellipse_fitting.u);
            ell.SetPoints(contours[n]);
            ell.ComputeAttributes();

            // Add those that meet the conditions to the list
            if (ell.minorLength / ell.majorLength >= axisRatio &&
                ell.majorLength > axisLength) { // Condition to add to ellipse list

                candidate_list.push_back(ell);
            }
        }
    }
    if (candidate_list.size() <= 1) return false;

    // Sort by longest axis of fitted ellipse
    std::sort(candidate_list.begin(), candidate_list.end(), compareEllipseSize);

    // Sort by longest axis of fitted ellipse
    Eigen::VectorXi use_index = Eigen::VectorXi::Ones(candidate_list.size());
    for (int n = 0; n < candidate_list.size() - 1; n++) {
        if (use_index(n) == 0) continue;

        Ellips target = candidate_list[n];

        for (int m = n + 1; m < candidate_list.size(); m++) {
            if (use_index(m) == 0) continue;

            Ellips reff = candidate_list[m];
            double dx = target.cx - reff.cx;
            double dy = target.cy - reff.cy;

            if ((dx * dx + dy * dy) < 4) {
                use_index(m) = 0;
            }
        }
    }
    ellipse_list.clear();
    for (int n = 0; n < candidate_list.size(); n++) {
        if (use_index(n) == 1) {
            ellipse_list.push_back(candidate_list[n]);
        }
    }

    // Draw detected ellipse centers
    //
    // In order to shorten the calculation time, it is better to comment out the following drawing process.
    if (drawEllipseCenter) {
        for (int n = 0; n < ellipse_list.size(); n++) {
            Ellips ell = ellipse_list[n];
            cv::Point p;
            p.x = ell.cx;
            p.y = ell.cy;
            cv::circle(image, p, 3, cv::Scalar(0, 255, 0), 1, 1);
        }
    }
    return true;
}

/* ****************************************** End of ellipse_detection.c *** */
