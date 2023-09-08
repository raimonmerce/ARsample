/* ****************************************************** metasequoia.h *** *
 * 3DモデルMetasequoiaクラス(ヘッダファイル)
 * ************************************************************************* */
#pragma once

#include "GLMetaseq.h"

class Metasequoia {
 public:
  // コンストラクタ
  Metasequoia();
  Metasequoia(char* filename);

  // デストラクタ
  ~Metasequoia ();

  // モデルを読み込む関数
  void OpenModel (char* filename);
  // モデルを描画する関数
  void DrawModel (void);
  
  // メンバ変数
  MQO_MODEL model;
  double    scale;

};

/* ************************************************ End of metasequoia.h *** */
