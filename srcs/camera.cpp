/*!
 * @file	camera.c
 * @brief	カメラでの映像撮影クラス
 */
#include "camera.h"

/*!
 * @brief  コンストラクタ
 */
CCamera::CCamera ()
{
  deviceID         = 0;
  inputMode        = INPUT_CAMERA;
  outputMode       = OUTPUT_WINDOW_AND_FILE;
  flipFlag         = false;
  undistortionFlag = false;
  inputFileName    = "";
  outputFileName   = "";
}

/*!
 * @brief コンストラクタ
 *
 * @param[in] _deviceID          デバイスID
 * @param[in] _inputMode         入力モード
 * @param[in] _outputMode        出力モード
 * @param[in] _flipFlag          入力を反転するかのフラグ
 * @param[in] _undistortionFlag  歪み補正をするかのフラグ
 */
CCamera::CCamera (int _deviceID,
		  int _inputMode,
		  int _outputMode,
		  bool _flipFlag,
		  bool _undistortionFlag)
{
  deviceID         = _deviceID;
  inputMode        = _inputMode;
  outputMode       = _outputMode;
  flipFlag         = _flipFlag;
  undistortionFlag = _undistortionFlag;
  inputFileName    = "";
  outputFileName   = "";
}

/*!
 * @brief  デストラクタ
 */
CCamera::~CCamera ()
{
  Close();
}

/*!
 * @brief　カメラのクローズ
 */
void CCamera::Close (void)
{
#ifdef USEPGR
  if (PGRCapture.IsConnected ()){
    FlyCapture2::Error error;
    error = mPGRCapture.StopCapture ();
    if (error != FlyCapture2::PGRERROR_OK){
      fprintf(stderr, "error at %d in %s\n", __LINE__, __FUNCTION__);
    }
    error = PGRCapture.Disconnect();
    if (error != FlyCapture2::PGRERROR_OK){
      fprintf(stderr, "error at %d in %s\n", __LINE__, __FUNCTION__);
    }
  }
#endif
  if (capture.isOpened ()) capture.release ();
}

/*!
 * @brief  カメラまたはビデオ画像をオープンする関数
 *
 * @param[in]	name		入力ビデオファイル名
 * @param[out]	width		ビデオ画像の横サイズ
 * @param[out]	height		ビデオ画像の縦サイズ
 * @param[out]	channels	ビデオ画像のチャネル数
 *
 * @retval	True or False
 */
bool CCamera::Open (int &w, int &h, int &c)
{
  if (inputMode == CCamera::INPUT_VIDEO) {
    return OpenVideo(inputFileName, w, h, c);
  }
  else {
    return OpenCamera(w, h, c);
  }
}

/*!
 * @brief　カメラのオープン
 *
 * @param[out]	w	画像の横サイズ
 * @param[out]	h	画像の縦サイズ
 * @param[out]	c	画像のチャネル数
 *
 * @retval  True or False   
 */
bool CCamera::OpenCamera (int &w, int &h, int &c)
{
  inputMode = CCamera::INPUT_CAMERA;
#ifdef USEPGR
  if (!PGRCapture.IsConnected ()){
    FlyCapture2::Error error;
    // カメラへの接続
    error = PGRCapture.Connect (0);
    if (error != FlyCapture2::PGRERROR_OK) {
      return false;
    }
    // カメラ情報の取得
    FlyCapture2::CameraInfo camInfo;
    error = PGRCapture.GetCameraInfo (&camInfo);
    if (error != FlyCapture2::PGRERROR_OK) {
      return false;
    }
    // 画像取得開始
    error = mPGRCap.StartCapture ();
    if (error != FlyCapture2::PGRERROR_OK) {
      return false;
    }
    // 画像の取得
    FlyCapture2::Image raw;
    error = PGRCapture.RetrieveBuffer (&raw);
    if (error != FlyCapture2::PGRERROR_OK) {
      return false;
    }
    // 画像サイズの取得
    w = width = raw.GetCols() / 2;
    h = height = raw.GetRows() / 2;
    c = channels = 3;
    imageSize = width * height * channels;
    image = cv::Mat (cv::Size (width, height), CV_8UC3);
		
    return true;
  }
  else {
    return false;
  }
#else
  if (!capture.isOpened ()) {
    // カメラデバイスのオープン
    if (deviceID != -1) {
      if (!capture.open (deviceID)) {
	fprintf (stderr, "Camera ID %d is not found\n", deviceID);
	return false;
      }
    } else {
      for (int n = 0; n < 5; n++) {
	if (!capture.open (n)) {
	  fprintf (stderr, "Camera ID %d is not found\n", n);
	} else {
	  fprintf (stdout, "Camera is opened on device ID %d\n", n);
	  deviceID = n;
	  break;
	}
      }
      if (deviceID == -1) return false;
    }
    // 画像サイズの設定
    capture.set (cv::CAP_PROP_FRAME_WIDTH, w);
    capture.set (cv::CAP_PROP_FRAME_HEIGHT, h);

    // 画像取得チェック
    int count = 0;
    while (image.data == NULL) {
      capture >> image;
      ++count;
      if (count > 10) {
	fprintf(stderr, "Cannot retrieve images\n");
	// 指定した画像サイズで画像を取得できなかった場合は、
	// 画像サイズをVGAにして再チェック
	w = 640;
	h = 480;
	capture.set (cv::CAP_PROP_FRAME_WIDTH, w);
	capture.set (cv::CAP_PROP_FRAME_HEIGHT, h);
	count = 0;
	while (image.data == NULL) {
	  capture >> image;
	  ++count;
	  if (count > 10){
	    fprintf(stderr, "Cannot retrieve images\n");
	    return false;
	  }
	}
      }
    }
    width = image.cols;
    height = image.rows;
    channels = c = image.channels();
    imageSize = height * (int) image.step;

    return true;
  } else {
    return false;
  }
#endif
}

/*!
 * @brief  入力ビデオのオープン
 *
 * @param[in]	name		入力ビデオファイル名
 * @param[out]	width		ビデオ画像の横サイズ
 * @param[out]	height		ビデオ画像の縦サイズ
 * @param[out]	channels	ビデオ画像のチャネル数
 *
 * @retval	True or False
 */
bool CCamera::OpenVideo (const std::string &name, int &w, int &h, int &c)
{
  inputMode = CCamera::INPUT_VIDEO;
  inputFileName = name;

  bool isOpen = capture.open (inputFileName);
  if (isOpen) {
    capture >> image;
    w = width = image.cols;
    h = height = image.rows;
    c = channels = image.channels ();
    imageSize = height * (int) image.step;
  }
  return isOpen;
}

/*! 
 * @brief  入力ビデオファイル名の登録
 *
 * @param[in] name  入力ビデオファイル名
 */
void CCamera::SetInputFile (const std::string& name)
{
  inputMode = INPUT_VIDEO;
  inputFileName = name;
}

/*! 
 * @brief  入力ビデオファイル名の取得
 * 
 * @return 入力ビデオファイル名
 */
std::string CCamera::GetInputFile (void)
{
  return inputFileName;
}

/*!
 * @brief　出力ビデオ画像の設定
 * @param[in]	name	出力ビデオファイル名
 * @param[in]	fps		フレームレート
 * @retval　True or Flase
 */
bool CCamera::SaveVideoSetting (const std::string &name, const float fps)
{
  bool isOpen;
#ifdef USEPGR
  isOpen = PGRCapture.IsConnected ();
#else
  isOpen = capture.isOpened ();
#endif
  if (isOpen) {	
    outputMode = OUTPUT_WINDOW_AND_VIDEO;
    outputFileName = name;
    bool isColor = (channels == 3) ? true : false;
    return writer.open (outputFileName,
			int (NULL), fps, cv::Size(width, height), isColor);
  } else {
    return false;
  }
}

/*!
 * @brief  画像の取得
 * @param[out]	img		取得した画像
 *
 * @retval	True or False
 */
bool CCamera::CaptureImage (void)
{
#ifdef USEPGR
  if (videoFlag) {
    capture >> image;
    if (image.empty ()) {
      capture.open (fileName);
      capture >> image;
    }
  } else if (PGRCapture.IsConnected ()) {
    FlyCapture2::Error error;
    FlyCapture2::Image raw;
    error = PGRCapture.RetrieveBuffer (&raw);
    if (error != FlyCapture2::PGRERROR_OK) {
      fprintf (stderr, "error at %d in %s\n", __LINE__, __FUNCTION__);
    }
    FlyCapture2::Image PGRimg;
    error = raw.Convert (FlyCapture2::PIXEL_FORMAT_BGR, &PGRimg);
    if (error != FlyCapture2::PGRERROR_OK) {
      fprintf (stderr, "error at %d in %s\n", __LINE__, __FUNCTION__);
    }
    cv::Mat cvImg(cv::Size(raw.GetCols (), raw.GetRows ()), CV_8UC3);
    int bufferSize
      = sizeof (unsigned char) * PGRimg.GetCols() * PGRimg.GetRows() * 3;
    memcpy_s (cvImg.data, bufferSize, PGRimg.GetData (), bufferSize);
    cv::resize (cvImg, image, image.size());
  } else {
    return false;
  }
#else
  if (capture.isOpened ()) {
    capture >> image;
    if (image.data == NULL) return false;
  } else {
    fprintf (stderr, "Failed to capture\n");
    return false;
  }
#endif
  if (undistortionFlag) {
    cv::remap (image.clone (), image, mapx, mapy, cv::INTER_LINEAR);
  }
  if (flipFlag) {
    cv::flip(image.clone(), image, 1);
  }
  return true;
}

/*!
 * @brief 画像の保存
 *
 * @param[in] name  出力画像ファイル名
 */
void CCamera::SaveImage (const std::string &name)
{
  if (outputMode == CCamera::OUTPUT_WINDOW_AND_FILE &&
      name != "") {
    cv::imwrite (name, image);
  } else {
    writer << image;
  }
}

/*!
 * @brief カメラパラメータの読み込み（カメラをオープンしてから呼び出す）
 *
 * @param[in]	name		  カメラパラメータファイル名
 * @param[in]	undistortion  画像の歪みを補正するかどうかのフラグ
 *
 * @retval	True or False
 */
bool CCamera::LoadParameters (const std::string &name,
			      const bool undistortion)
{
  cv::FileStorage fs;
  if (!fs.open (name, cv::FileStorage::READ)) {
    fprintf (stderr, "Cannot load camera parameters\n");
    return false;
  }
  //cv::FileNode node (fs.fs, NULL);

  //cv::read(node["A"], internalParams);
  //cv::read(node["D"], distortionParams);

  undistortionFlag = undistortion;

  if (undistortionFlag) {
    cv::initUndistortRectifyMap (internalParams,
				 distortionParams,
				 cv::Mat (),
				 internalParams,
				 image.size (),
				 CV_32FC1, mapx, mapy);
    distortionParams = cv::Mat_<double>::zeros(5, 1);
  }
  return true;
}

/*!
 * @brief  カメラパラメータの保存
 *
 * @param[in]	name  カメラパラメータを保存するファイル名
 *
 * @retval  True of False
 */
bool CCamera::SaveParameters (const std::string &name) const
{
  if (internalParams.empty () || distortionParams.empty ()) {
    fprintf (stderr, "Empty parameters\n");
    return false;
  }
  cv::FileStorage fs;
  fs.open (name + ".xml", cv::FileStorage::WRITE);
  cv::write (fs, "A", internalParams);
  cv::write (fs, "D", distortionParams);

  fs.release();

  return true;
}

/*!
 * @breif	画像の横サイズの取得
 * 
 * @retval	画像の横サイズ
 */
int CCamera::GetImageWidth(void)
{
  return width;
}

/*!
 * @breif	画像の縦サイズの取得
 *
 * @retval	画像の縦サイズ
 */
int CCamera::GetImageHeight(void)
{
  return height;
}

/*! 
 * @breif  カメラデバイスIDの設定
 *
 * @param[in] _deviceID  カメラのデバイスID
 */
void CCamera::SetDeviceID(int _deviceID)
{
  deviceID = _deviceID;
  std::cout << "Set devide ID to " << deviceID << std::endl;

}

/*!
 * @brief  カメラデバイスIDの取得
 *
 * retval  カメラのデバイスID
 */
int  CCamera::GetDeviceID(void)
{
  return deviceID;
}
