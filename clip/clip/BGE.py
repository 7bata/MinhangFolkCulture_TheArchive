from FlagEmbedding import FlagAutoModel
import pandas as pd
# 加载 BGE 模型
model = FlagAutoModel.from_finetuned('BAAI/bge-base-zh-v1.5',
                                     query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                     use_fp16=True)
# 读取数据
df = pd.read_excel('metadata_revised_test.xlsx')
df['embedding'] = df['embedding'].astype(object)

for index, value in df["link"].items():
    if df["category"][index] == "文档" and pd.notna(value):
        embedding = model.encode(value)
        try:
            df.at[index, "embedding"] = embedding.tolist()
            print(str(index+1)+"行数据embedding储存成功")
        except:
            print(str(index+1)+"行数据embedding储存失败")
df.to_excel('metadata_revised_test.xlsx', index=False)
print("complete")
