/* ******************************************************* application.h *** *
 * アプリケーションクラス
 * ************************************************************************* */
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "camera.h"
#include "glfw_window.h"
#include "ellipse.h"
#include "ellipse_detection.h"
#include "circular_marker_detection.h"
#include "circular_marker.h"
#include "rectangular_marker.h"
#include "metasequoia.h"

typedef struct _GLProjectionParam {
  double horiz;
  double vert;
  double nearDist;
  double farDist;
} GLProjectionParam;

class Application {
public:
  // コンストラクタ
  Application(const char* title);

  // デストラクタ
  ~Application();

  // カメラ関係の関数
  void ReadSettings(const std::string& filename);
  bool OpenCamera(void);

  // マーカーを検出する関数
  bool MarkerDetect(void);
  bool MarkerDetectRect(void);
  
  // 定数
  const int DRAW_INPUT      = 0;
  const int IMAGE_WIDTH     = 640;
  const int IMAGE_HEIGHT    = 480;  
  const int IMAGE_U0        = 320;
  const int IMAGE_V0        = 240;
  const double CAMERA_FOCUS = 700.0;
  const double DEFAULT_SCALE = 0.005;
  const double DEFAULT_FAR_SCALE = 1.0e+6;
  
  // メンバ変数
  CCamera camera;    // カメラ
  GLFWWindow window; // 表示用ウィンドウ
  cv::Mat image;     // 背景用画像
  int drawMode;      // 何を描画するか
  double focus;      // カメラの焦点距離
  double u0;         // 光軸点の座標(u0, v0)
  double v0;
  Eigen::MatrixXd A; // 楕円パラメータの座標系変換行列
  GLProjectionParam proj_param;
  
  EllipseDetection            ellipse_detector; // 楕円検出クラス
  CircularMarkerDetection     marker_detector;  // マーカー検出クラス
  std::vector<Ellips>        ellipse_list;     // 楕円リスト
  std::vector<std::vector<cv::Point>>        rectangle_list;     // 

  std::vector<CircularMarker> marker_list;      // マーカーリスト
  std::vector<RectangularMarker> marker_list_rect;

  Metasequoia model;
  char model_filename[1024];
  double model_scale;
};

/* ************************************************ End of application.h *** */
