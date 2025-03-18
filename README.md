# 目的
風景をタッチした部位に応じて音を鳴らし分けたい。  
・領域が音の種類を決定  
・深度が音の音量を決定  
このようなゲームをpygameで実装する。

# 手順
assetsのimageの中に風景の画像を保存する  
assetsのsceneの中に風景の動画を保存する  
assetsのsoundの中に風景の動画から音のみを抽出したものを保存する  

（それぞれのコード中の該当部分をそのアセットパスに変更する）  
python model/semantic_segmentation.py　を実行（segmentation.jsonに領域情報が保存される）  
python model/estimate_depth.py　を実行（depth_data.jsonに深度情報が保存される）  

（必要に応じてpython model/class_check.py　を実行し、領域に対応する番号を編集する）  
python app/main.pyを実行（ゲームをプレイ）  

# AI関連技術
領域分類にはNvidiaのSegformerモデルを使用  
深度推定にはIntelのMiDaSモデルを使用  

# アセット
Atmoph公式サイトのPreview画像や動画、効果音ラボのSEを拝借している。
