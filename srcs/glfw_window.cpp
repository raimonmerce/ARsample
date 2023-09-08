/* ******************************************************* glfw_window.c *** *
 * GLFWウィンドウクラス
 * ************************************************************************* */
#include "glfw_window.h"

/*
 * 標準のキーボード関数
 */
static void _keyFunc (GLFWwindow*	window,
		      int		key,
		      int		scancode,
		      int		action,
		      int		mods) {
  if (key == GLFW_KEY_Q) {
    glfwTerminate();
    exit(1);
  }
}

/*
 * 標準コンストラクタ
 */
GLFWWindow::GLFWWindow() {
  if (glfwInit() == GL_FALSE) {
    fprintf (stderr, "GLFW initialization failed.\n");
  }
}

/*
 * ウィンドウ生成まで行うコンストラクタ
 *
 * @param [in] width  : ウィンドウの横サイズ
 * @param [in] height : ウィンドウの縦サイズ
 * @param [in] title  : ウィンドウタイトル
 */
GLFWWindow::GLFWWindow(int width, int height, char* title) {
  if (glfwInit() == GL_FALSE) {
    fprintf (stderr, "GLFW initialization failed.\n");
  } else {
    window = glfwCreateWindow (width, height, title, NULL, NULL);
    glfwSetKeyCallback(window, _keyFunc);
    glfwMakeContextCurrent (window);
  }
}

/*
 * デストラクタ
 */
GLFWWindow::~GLFWWindow() {
  glfwTerminate();
}

/*
 * ウィンドウを生成する関数
 *
 * @param [in] width  : ウィンドウの横サイズ
 * @param [in] height : ウィンドウの縦サイズ
 * @param [in] title  : ウィンドウタイトル
 */
void GLFWWindow::MakeContext (int width, int height, const char* title) {
  window = glfwCreateWindow (width, height, title, NULL, NULL);
  glfwSetKeyCallback(window, _keyFunc);
  glfwMakeContextCurrent (window);
}

/*
 * 画像を描画する関数
 *
 * @param [in] pixels : 画像データ
 * @param [in] width  : ウィンドウの横サイズ
 * @param [in] height : ウィンドウの縦サイズ
 */
void GLFWWindow::drawImage (GLubyte* pixels, int image_width, int image_height, int window_width, int window_height) {
  glDisable (GL_DEPTH_TEST);

  glMatrixMode (GL_MODELVIEW);
  glPushMatrix ();
  glLoadIdentity ();

  glMatrixMode (GL_PROJECTION);
  glPushMatrix ();
  glLoadIdentity ();

  glViewport(0, 0, window_width, window_height);
  glOrtho (0.0, (GLdouble) image_width, 0.0, (GLdouble) image_height, -1.0, 1.0);
  glPixelZoom ((double) window_width / image_width, (double) -window_height / image_height);
  glRasterPos2i (0, image_height - 1);
  glDrawPixels (image_width, image_height, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);

  glMatrixMode (GL_PROJECTION);
  glPopMatrix ();
  glMatrixMode (GL_MODELVIEW);
  glPopMatrix ();

  glEnable (GL_DEPTH_TEST);
}

/*
 * 世界座標系を描画する関数
 */
void GLFWWindow::drawWAxis(void) {
  glBegin(GL_LINES);
  {
    glColor4f(1.0, 0.0, 0.0, 1.0);
    glVertex3d(300.0, 0.0, 0.0);
    glVertex3d(0.0, 0.0, 0.0);

    glColor4f(0.0, 1.0, 0.0, 1.0);
    glVertex3d(0.0, 300.0, 0.0);
    glVertex3d(0.0, 0.0, 0.0);

    glColor4f(0.0, 0.0, 1.0, 1.0);
    glVertex3d(0.0, 0.0, 300.0);
    glVertex3d(0.0, 0.0, 0.0);
  }
  glEnd();
}

/*
 * キーボード関数を登録する関数
 */
void::GLFWWindow::SetKeyFunc(void (*KeyFunc) (GLFWwindow*	window,
					      int		key,
					      int		scancode,
					      int		action,
					      int		mods)) {
  glfwSetKeyCallback(window, KeyFunc);
}

/* ************************************************ End of glfw_window.c *** */
