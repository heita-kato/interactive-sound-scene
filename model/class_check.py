import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torchvision.transforms.functional as F

# クラスラベル（ADE20K）
ADE20K_CLASSES = [
    "background", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "window", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table",
    "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence",
    "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "sign", "chest", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool", "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair",
    "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck",
    "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt", "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle",
    "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot",
    "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
    "hood", "sconce", "vase", "traffic light", "tray", "trash can", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower",
    "radiator", "glass", "clock", "flag"
]

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
    Image.fromarray(segmentation_map.astype(np.uint8)),
    size=(original_size[1], original_size[0]), interpolation=Image.NEAREST
)
segmentation_map_resized_array = np.array(segmentation_map_resized)

# 可視化
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 元画像
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# セグメンテーションマップ
im = ax[1].imshow(segmentation_map_resized, cmap="jet")
ax[1].set_title("Segmentation Result")
ax[1].axis("off")


# クリックイベント
def on_click(event):
    if event.inaxes == ax[1]:  # セグメンテーションマップ側でのみ有効
        x, y = int(event.xdata), int(event.ydata)
        class_idx = segmentation_map_resized_array[y, x]
        class_name = ADE20K_CLASSES[class_idx] if class_idx < len(ADE20K_CLASSES) else "Unknown"
        print(f"Clicked at ({x}, {y}) → Class: {class_name} ({class_idx})")


cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()

# 使用例
#python model/class_check.py