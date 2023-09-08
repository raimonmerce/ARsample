/*!
 * @file	camera.h
 * @brief	カメラクラス
 */
#pragma once

#include <opencv2/opencv.hpp>

#ifdef USEPGR
#include <FlyCapture2.h>
#endif

/*!
 * @class  カメラクラス
 * @brief　カメラクラス
 */
class CCamera
{
  // メソッド
 public:
  // コンストラクタ
  CCamera ();
  CCamera (int _deviceID,
	   int _inputMode,
	   int _outputMode,
	   bool _flipFlag,
	   bool _undistortionFlag = false);
		
  // デストラクタ
  ~CCamera ();

  void SetDeviceID (int _deviceID);
  int  GetDeviceID (void);

  // 入力設定
  bool Open (int &w, int &h, int &c);
  bool OpenCamera (int &w, int &h, int &c);
  bool OpenVideo (const std::string &name, int &w, int &h, int &c);

  // カメラのクローズ
  void Close (void);

  // 画像の取得
  bool CaptureImage (void);

  // 画像の保存
  void SaveImage(const std::string &name);

  // 入力ビデオファイル名の登録
  void SetInputFile(const std::string& name);
  std::string GetInputFile (void);

  // ビデオ出力の設定
  bool SaveVideoSetting(const std::string &name, const float fps = 30.f);

  // カメラパラメータ関連
  bool LoadParameters (const std::string &name,
		       const bool undistortion = true);
  
  bool SaveParameters (const std::string &name) const;

  // 画像サイズの取得関数
  int GetImageWidth (void);
  int GetImageHeight (void);

  // メンバー
 public:
  // 入力モード
  static const int INPUT_CAMERA = 0;
  static const int INPUT_VIDEO  = 1;

  // 出力モード
  static const int OUTPUT_WINDOW_AND_FILE  = 0;
  static const int OUTPUT_WINDOW_AND_VIDEO = 1;
  
  int     inputMode;        // 入力モード（0: カメラ, 1: ビデオ）
  int     outputMode;       // 出力モード（0: 静止画, 1: 動画）
  cv::Mat image;            // 画像データ
  bool    flipFlag;         // 画像の水平反転を行うかどうか
  bool    undistortionFlag; // 歪み補正を行うかどうか

#ifdef USEPGR
  // ポイントグレイ製カメラを使用する場合
  FlyCapture2::Camera PGRCapture;
#endif
  cv::VideoCapture capture;
  int deviceID;               // デバイスID
		
  // 画像の歪み補正関連
  cv::Mat mapx;               // 歪み補正のための情報
  cv::Mat mapy;               // 歪み補正のための情報
  cv::Mat internalParams;     // カメラの内部パラメータ
  cv::Mat distortionParams;   // 歪み補正パラメータ

  // 画像サイズ関連
  int width;                  // 画像の幅
  int height;                 // 画像の高さ
  int channels;               // チャネル（プレーン）数
  int imageSize;              // 画像データサイズ (width*height*channels)

  // ビデオ出力関連
  cv::VideoWriter writer;     // OpenCVのビデオ出力クラス

  std::string inputFileName;  // 入力ビデオファイル名
  std::string outputFileName; // 出力ビデオファイル名
};
