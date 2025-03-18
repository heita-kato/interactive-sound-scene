import json
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt

def estimate_depth(image_path, json_output_path, depth_output_path):
    # MiDaSモデルのロード
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    
    # 画像の読み込み
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # 入力画像の元のサイズ (width, height)
    
    # 変換処理の定義
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 入力画像を384x384にリサイズ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 画像の前処理
    input_tensor = transform(image).unsqueeze(0)
    
    # 深度推定
    with torch.no_grad():
        depth_map = model(input_tensor)
    
    # NumPy配列に変換
    depth_map = depth_map.squeeze().cpu().numpy()
    
    # 深度マップのリサイズ（元の画像サイズに戻す）
    depth_map_resized = cv2.resize(depth_map, (original_size[0], original_size[1]), interpolation=cv2.INTER_CUBIC)
    
    # 画像サイズを取得
    height, width = depth_map_resized.shape
    
    # 深度情報をJSONに変換
    depth_data = {
        "width": width,
        "height": height,
        "map": depth_map_resized.tolist()
    }
    
    # JSONファイルに保存
    with open(json_output_path, "w") as json_file:
        json.dump(depth_data, json_file, indent=4)
    
    print(f"Depth data saved to {json_output_path}")
    
    # 深度マップの可視化
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_map_resized, cmap='plasma')  # plasmaカラーマップで深度を可視化
    plt.colorbar(label="Depth")
    plt.axis("off")
    plt.show()
    
    print(f"Depth visualization saved to {depth_output_path}")

estimate_depth("assets\image\schwarzsee_and_ober_gabelhorn_m.jpg", "db\depth_data.json", "depth_visualization.png")

# 使用例
#python model/estimate_depth.py