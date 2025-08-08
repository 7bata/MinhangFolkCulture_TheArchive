import re
import torch
import clip
from PIL import Image
import pandas as pd

df = pd.read_excel('metadata_revised_test.xlsx')
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for index, value in df["link"].items():
    if df["category"][index] == "图片" and pd.notna(value):
        match = re.search(r'"([^"]*)"', value)
        image = Image.open("E:/Project/MinhangFolkCulture_TheArchive/MinhangFolkCulture_TheArchive" + match.group(1)+".jpg")
        image_input = preprocess(image).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                df.at[index, "embedding"] = image_features.cpu().detach().numpy().tolist()[0]
                print("Success"+str(index))

        except:
            print("Error"+str(index))
df.to_excel('metadata_revised_test.xlsx', index=False)

print("complete")


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
