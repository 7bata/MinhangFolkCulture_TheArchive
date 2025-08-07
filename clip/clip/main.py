import torch
import clip
from PIL import Image




device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载图像
image = Image.open("/Users/tbata/MFC_Project/MinhangFolkCulture_TheArchive/clip/clip/001手狮舞.jpg")
 
# 应用CLIP的预处理变换
image_input = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备

# 单个文本提示词
texts = ["一张手狮舞的照片", "一只在奔跑的狗", "夜晚的城市风景"]
text_input = clip.tokenize(texts).to(device)

# 获取预测
with torch.no_grad():
    # 计算特征
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)
    
print("Image features shape:", image_features)
print("Text features shape:", text_features)