/* ******************************************************* application.c *** *
 * アプリケーションクラス
 * ************************************************************************* */
#include "application.h"
#include <fstream>
/*!
 * @brief コンストラクタ
 */
Application::Application(const char* title) {
  // カメラに関する変数
  camera.deviceID = 0;
  camera.width    = IMAGE_WIDTH;
  camera.height   = IMAGE_HEIGHT;
  
  focus    = CAMERA_FOCUS; // [画素]
  u0       = IMAGE_U0;     // [画素]
  v0       = IMAGE_V0;     // [画素]  

  proj_param.horiz = (camera.width  / 2.0) * DEFAULT_SCALE;
  proj_param.vert  = (camera.height / 2.0) * DEFAULT_SCALE;
  proj_param.nearDist  = focus * DEFAULT_SCALE;
  proj_param.farDist   = proj_param.nearDist * DEFAULT_FAR_SCALE;
    
  // ウィンドウに関数変数
  window.MakeContext (camera.width, camera.height, title);
  
  // ウィンドウに描画する背景画像
  drawMode = DRAW_INPUT;

  // 座標系の変換行列
  A = Eigen::MatrixXd::Zero(3, 3);
  A << 0.0, focus, u0, -focus, 0.0, v0, 0.0, 0.0, 1.0;
  marker_detector.A = A;
  marker_detector.drawMarker = true;
  ellipse_detector.drawEllipseCenter = true;
}

/*!
 * @brief　デストラクタ
 */
Application::~Application() {
  camera.Close();
}

/*!
 * @brief  設定ファイルの読み込み
 *
 * @param[in] filename  設定ファイル名
 */
void Application::ReadSettings(const std::string& filename)
{
  std::ifstream ifs(filename);
  if (!ifs.fail()) {
    ifs >> camera.deviceID;
    ifs >> camera.width >> camera.height;
    ifs >> focus;
    ifs >> model_filename;
    ifs >> model.scale;
    ifs >> marker_detector.radiusOuter;
    ifs >> marker_detector.radiusInner;
    ifs.close();
  } else {
    std::cout << "Setting file open error." << std::endl;
  }
  u0 = camera.width  / 2.0;
  v0 = camera.height / 2.0;
  A << 0.0, focus, u0, -focus, 0.0, v0, 0.0, 0.0, 1.0;
  
  proj_param.horiz = (camera.width  / 2.0) * DEFAULT_SCALE;
  proj_param.vert  = (camera.height / 2.0) * DEFAULT_SCALE;
  proj_param.nearDist  = focus * DEFAULT_SCALE;
  proj_param.farDist   = proj_param.nearDist * DEFAULT_FAR_SCALE;

  glfwSetWindowSize (window.window, camera.width, camera.height);
  glViewport(0, 0, camera.width, camera.height);  
}

/*!
 * @brief  カメラのオープン
 *
 * @retval  True of False
 */
bool Application::OpenCamera(void)
{
  int channel = 3;
  return camera.Open(camera.width, camera.height, channel);
}

/*!
 * @brief 楕円検出
 */
bool Application::MarkerDetect(void) {
  camera.CaptureImage();
  bool retval = ellipse_detector.Detect(camera.image, ellipse_list);
  //bool retval = ellipse_detector.DetectRect (camera.image, rectangle_list);  
  return retval;
  if (retval) {
    retval = marker_detector.Detect (ellipse_list, camera.image, marker_list);
    //retval = marker_detector.DetectRect(rectangle_list, camera.image, marker_list);
  }
  
  return retval;
}

bool Application::MarkerDetectRect(void) {
    camera.CaptureImage();
    bool retval = ellipse_detector.DetectRect (camera.image, rectangle_list, marker_list_rect, A);
   
    return retval;
}

/* ************************************************ End of application.c *** */
