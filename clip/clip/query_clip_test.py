import re
import torch
import clip
from PIL import Image
import pandas as pd

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = "河清竹编篮是什么？"
with torch.no_grad():
    try:
        query_input = clip.tokenize(text).to(device)
        print("未截断")
    except RuntimeError:
        query_input = clip.tokenize(text, truncate=True).to(device)
        print("截断")
text_features = model.encode_text(query_input)
print(text_features.cpu().detach().numpy().tolist()[0])