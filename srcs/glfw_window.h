/* ******************************************************* glfw_window.h *** *
 * GLFW・ヲ・」・ノ・ヲ・ッ・鬣ケ(・リ・テ・タ・ユ・。・、・・
 * ************************************************************************* */
#pragma once

#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
//#include <glut.h>

class GLFWWindow
{
 public:
  // ・ウ・ケ・ネ・鬣ッ・ソ
  GLFWWindow();
  GLFWWindow(int width, int height, char* title);

  // ・ヌ・ケ・ネ・鬣ッ・ソ
  ~GLFWWindow();

  void MakeContext (int width, int height, const char* title);
  void drawImage (GLubyte* pixels, int image_width, int image_height, int window_width, int window_height);
  void drawWAxis (void);
  void SetKeyFunc (void (*KeyFunc) (GLFWwindow*	window,
				    int		key,
				    int		scancode,
				    int		action,
				    int		mods));

  GLFWwindow* window;
};

/* ************************************************ End of glfw_window.h *** */

