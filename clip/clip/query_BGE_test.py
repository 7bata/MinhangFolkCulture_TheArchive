from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-zh-v1.5',
                                     query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                     use_fp16=True)
query = "宣卷有着怎么样的历史？"
embedding = model.encode(query)
print(embedding.tolist())
