import torch
import clip
from PIL import Image
import pandas as pd


df = pd.read_excel('metadata.xlsx')
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for index, value in df["description"].items():
    if df["category"][index] == "文档":
        doc = value
        text_input = clip.tokenize(doc).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            print(text_features.shape)




'''
# 加载图像
image = Image.open("001手狮舞.jpg")
 
# 应用CLIP的预处理变换
image_input = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备

# 单个文本提示词
texts = ["一张手狮舞的照片", "一只在奔跑的狗", "夜晚的城市风景"]


# 获取预测

    # 计算特征
    image_features = model.encode_image(image_input)
'''
    
