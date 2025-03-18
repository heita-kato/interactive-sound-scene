import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torchvision.transforms.functional as F

# モデルのロード
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

# 画像の読み込み
image_path = "assets/image/schwarzsee_and_ober_gabelhorn_m.jpg"
image = Image.open(image_path).convert("RGB")
original_size = image.size  # (width, height)

# 前処理
inputs = feature_extractor(images=image, return_tensors="pt")

# 推論
with torch.no_grad():
    outputs = model(**inputs)

# セグメンテーション結果
logits = outputs.logits
segmentation_map = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

# 元の画像サイズにリサイズ
segmentation_map_resized = F.resize(
    Image.fromarray(segmentation_map.astype(np.uint8)), size=(original_size[1], original_size[0]), interpolation=Image.NEAREST
)

# 可視化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmentation_map_resized, cmap="jet", alpha=0.7)
plt.title("Segmentation Result")
plt.axis("off")

plt.show()

# JSON に保存
segmentation_map_resized_array = np.array(segmentation_map_resized)
segmentation_data = {
    "width": original_size[0],
    "height": original_size[1],
    "map": segmentation_map_resized_array.tolist()
}

with open("db/segmentation.json", "w") as json_file:
    json.dump(segmentation_data, json_file, indent=4)

print("Segmentation map saved to segmentation.json")

# 使用例
#python model/semantic_segmentation.py