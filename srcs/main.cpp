#include <opencv2/opencv.hpp>
#include <math.h>
#include <string.h>
#include "application.h"

static void display (Application& app) {
  // 画面のクリア
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /* ****************************************************************** *
   * アプリケーションに応じた処理を書く(ここから)
   * ****************************************************************** */
  /* マーカー検出 */
  //bool detected = app.MarkerDetect();
  bool detected = app.MarkerDetectRect();
  // 画像の描画
  int window_width, window_height;
  glfwGetWindowSize(app.window.window, &window_width, &window_height);
  app.window.drawImage ((GLubyte *) app.camera.image.data, app.camera.width, app.camera.height, window_width, window_height);

  /* ****************************************************************** *
   * アプリケーションに応じた処理を書く(ここまで)
   * ****************************************************************** */

  // 3Dモデルの描画
  if (detected) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-app.proj_param.horiz, app.proj_param.horiz,
	      -app.proj_param.vert, app.proj_param.vert,
	      app.proj_param.nearDist, app.proj_param.farDist);

    //for (int n = 0; n < (int) app.marker_list.size(); n++) {
    //for (int n = 0; n < (int)app.marker_list_rect.size(); n++) {
    if (0 < (int)app.marker_list_rect.size()) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      /* ************************************************************* *
       * Drawing part of 3D model
       * ************************************************************* */
      /* Calculate camera parameters from detected markers */
      //app.marker_list[n].ComputeCameraParam();
      app.marker_list_rect[0].ComputeCameraParam();
      /* Register modelview matrix */
      //glLoadMatrixf(app.marker_list[n].M);
      glLoadMatrixf(app.marker_list_rect[0].M);
      /* drawing the model */
      app.model.DrawModel();
      /* ************************************************************* *
       * Drawing part of 3D model
       * ************************************************************* */
    }
  }
  glfwSwapBuffers (app.window.window);
}

static void CustomKeyFunc (GLFWwindow*	window,
			   int		key,
			   int		scancode,
			   int		action,
			   int		mods) {
  static int count = 0;

  if (key == GLFW_KEY_Q) {
    glfwTerminate();
    exit(1);
  } else if (key == GLFW_KEY_S) {
    unsigned char* imageBuffer = NULL;
    glReadBuffer (GL_BACK);
    cv::Mat image(cv::Size(640, 480), CV_8UC4);
    glReadPixels (0, 0,
		  640, 480,
		  GL_BGRA_EXT,
		  GL_UNSIGNED_BYTE,
		  image.data);
    char name[1024];
    sprintf_s (name, "output-%03d.png", count++);
    cv::flip (image ,image, 0);
    cv::imwrite (name, image);
  }
}

int main (int argc, char **argv)
{
  /* アプリケーションクラスの作成 */
  Application app("Circular Marker Demo");
  
  /* カメラの設定 */
  app.ReadSettings ("./settings.txt");
  if (!app.OpenCamera ()) exit (1);

  /* 3Dモデルの読み込み */
  app.model.OpenModel(app.model_filename);  

  app.window.SetKeyFunc (CustomKeyFunc);
  
  /* メインループ */
  while (!glfwWindowShouldClose (app.window.window)) {
    display (app);
    glfwWaitEventsTimeout(1e-03);
  }
  return 0;
}
