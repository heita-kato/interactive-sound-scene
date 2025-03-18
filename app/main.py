import cv2
import json
import random
import numpy as np
import pygame

# 音声初期化
pygame.mixer.init()

# segの種類に対応する音ファイル
seg_sounds = {
    2: pygame.mixer.Sound('assets/sound/blue_sky.mp3'),  # sky
    13: pygame.mixer.Sound('assets/sound/grass_sound.mp3'),  # grass
    16: pygame.mixer.Sound('assets/sound/mountain_rocks.mp3'),  # mountain
    21: pygame.mixer.Sound('assets/sound/water_sound.mp3'),  # pond
    76: pygame.mixer.Sound('assets/sound/building_sound.mp3'),  # building
    94: pygame.mixer.Sound('assets/sound/grass_sound.mp3')  # grass
}

# 動画ファイルのパス
video_path = 'assets/scene/swizerland_tiktok.mp4'
cap = cv2.VideoCapture(video_path)

# 音声をpygameで再生
pygame.mixer.music.load('assets/sound/swizerland_tiktok.mp3')
pygame.mixer.music.play(loops=0, start=0.0)

# JSONデータ読み込み
with open('db/segmentation.json', 'r') as f:
    segmentation_json_data = json.load(f)
segmentation_map = segmentation_json_data["map"]

with open('db/depth_data.json', 'r') as f:
    depth_json_data = json.load(f)
depth_map = depth_json_data["map"]

# sparkleエフェクト
sparkle_points = []
SPARKLE_LIFETIME = 15

# 表示サイズとjsonサイズ
display_width = 500
display_height = 888
json_width = 250
json_height = 444

# クリックイベント
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 表示サイズ → jsonサイズにスケーリング
        scaled_x = int(x * json_width / display_width)
        scaled_y = int(y * json_height / display_height)

        seg_val = segmentation_map[scaled_y][scaled_x]
        dep_val = depth_map[scaled_y][scaled_x]

        # sparkle座標は表示サイズで
        sparkle_points.append({'pos': (x, y), 'lifetime': SPARKLE_LIFETIME})

        # ---- サウンド処理 ----
        if seg_val in seg_sounds:
            sound = seg_sounds[seg_val]

            if seg_val == 2:
                volume = 1.0
            else:
                # depの値を正規化
                volume = max(0.1, min(1.0, dep_val / 1000))
                sound.set_volume(volume)
            sound.play()

            print(f"Clicked at: x={x}, y={y}, seg={seg_val}, dep={dep_val}, vol={volume}")

# ウィンドウ設定
cv2.namedWindow("Interactive Sound Test")
cv2.setMouseCallback("Interactive Sound Test", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームリサイズ
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # sparkle描画
    sparkle_layer = np.zeros_like(resized_frame, dtype=np.uint8)
    for sparkle in sparkle_points:
        x, y = sparkle['pos']
        alpha = sparkle['lifetime'] / SPARKLE_LIFETIME
        sparkle_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        for _ in range(10):
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-5, 5)
            cv2.circle(sparkle_layer, (x + offset_x, y + offset_y), 1, sparkle_color, -1)
        sparkle['lifetime'] -= 1

    blended = cv2.addWeighted(resized_frame, 1.0, sparkle_layer, 1.0, 0)
    sparkle_points = [s for s in sparkle_points if s['lifetime'] > 0]

    # 表示
    cv2.imshow("Interactive Sound Test", blended)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

#実行例('q'で終了)
#python app/main.py