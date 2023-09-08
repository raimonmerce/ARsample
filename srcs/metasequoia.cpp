/* ****************************************************** metasequoia.c *** *
 * 3DモデルMetasequoiaクラス
 * ************************************************************************* */
#include "metasequoia.h"

/*
 * コンストラクタ
 */
Metasequoia::Metasequoia() {
  scale = 1.0;
}

/*
 * コンストラクタ
 */
Metasequoia::Metasequoia(char* filename) {
  scale = 1.0;  
  model = mqoCreateModel (filename, scale);
}

/*
 * デストラクタ
 */
Metasequoia::~Metasequoia() {
  ;
}

/*
 * 3Dモデルの読み込み関数
 *
 * @param [in] filename : ファイル名
 */
void Metasequoia::OpenModel (char* filename) {
  model = mqoCreateModel (filename, scale);
}

/*
 * 3Dモデルの描画関数
 */
void Metasequoia::DrawModel (void) {
  mqoCallModel (model);
}

/* ************************************************ End of metasequoia.c *** */
