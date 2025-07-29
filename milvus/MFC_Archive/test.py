import pandas as pd
from pymilvus import MilvusClient, model
import ast # 用于安全地解析字符串形式的列表

# --- 1. 初始化 Milvus 客户端和嵌入模型 ---
# 这部分保留了源程序的核心设置
print("正在初始化 Milvus 客户端和嵌入模型...")
client = MilvusClient("doc_similarity.db")
embedding_fn = model.DefaultEmbeddingFunction()

COLLECTION_NAME = "doc_comparison_collection"

# --- 2. 加载并准备数据 ---
print("正在加载 CSV 文件 'test_data_des_part2.xlsx'...")
try:
    df = pd.read_excel('test_data_des_part2.xlsx')
    print("文件加载成功。")
except FileNotFoundError:
    print("错误: 'test_data_des_part2.xlsx' 文件未找到。请确保文件与脚本在同一目录下。")
    exit()

# 用于存储最终结果的列表
results_list = []

# --- 3. 循环处理每一行数据 ---
print(f"开始处理文件中的 {len(df)} 行数据...")
for index, row in df.iterrows():
    print(f"\n正在处理第 {index + 1} 行...")

    # 获取原始数据
    doc_original = str(row['doc_original'])
    doc_revised = str(row['doc_revised'])
    query_vector_str = row['query_vector']

    # 安全地将字符串转换为关键词列表
    try:
        keywords = ast.literal_eval(query_vector_str)
        if not isinstance(keywords, list):
            print(f"  - 警告: 第 {index + 1} 行的 query_vector 不是一个列表，将跳过此行。")
            continue
    except (ValueError, SyntaxError):
        print(f"  - 警告: 无法解析第 {index + 1} 行的 query_vector，将跳过此行。内容: {query_vector_str}")
        continue
        
    # --- 4. Milvus 集合管理 (每行一个独立的集合) ---
    # 这确保了每次比较的纯净性，遵循了源程序的逻辑
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=embedding_fn.dim,
    )

    # --- 5. 嵌入并插入文档 ---
    docs_to_embed = [doc_original, doc_revised]
    vectors = embedding_fn.encode_documents(docs_to_embed)

    data_to_insert = [
        {"id": 0, "vector": vectors[0], "doc_name": "original"},
        {"id": 1, "vector": vectors[1], "doc_name": "revised"}
    ]
    
    client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"  - 已将 'doc_original' 和 'doc_revised' 插入 Milvus 集合。")

    # --- 6. 对每个关键词进行相似度搜索 ---
    print(f"  - 正在与 {len(keywords)} 个关键词进行比较...")
    for keyword in keywords:
        query_vectors = embedding_fn.encode_queries([keyword])
        print(keyword)
        # 执行搜索
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            data=query_vectors,
            limit=2,  # 返回与原始和修订文档的两个匹配结果
            output_fields=["doc_name"],
        )
        print(search_res)
        # 解析搜索结果
        similarity_original = None
        similarity_revised = None

        # Milvus 返回的是距离 (distance)，对于 cosine 相似度, similarity = 1 - distance
        for hit in search_res[0]:
            similarity_score = 1 - hit['distance']
            if hit['entity']['doc_name'] == 'original':
                similarity_original = similarity_score
            elif hit['entity']['doc_name'] == 'revised':
                similarity_revised = similarity_score
        
        # 将当次结果存入列表
        results_list.append({
            'row_id': index + 1,
            'keyword': keyword,
            'similarity_original': f"{similarity_original:.4f}" if similarity_original is not None else "N/A",
            'similarity_revised': f"{similarity_revised:.4f}" if similarity_revised is not None else "N/A",
            'doc_original': doc_original,
            'doc_revised': doc_revised,
        })

# --- 7. 清理并导出结果 ---
print("\n所有行处理完毕。正在清理资源并导出结果...")

# 删除最后使用的集合
if client.has_collection(collection_name=COLLECTION_NAME):
    client.drop_collection(collection_name=COLLECTION_NAME)

client.close()

# 将结果列表转换为 DataFrame
results_df = pd.DataFrame(results_list)

# 导出到 CSV 文件
output_filename = 'similarity_comparison_results.csv'
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n处理完成！对比结果已成功导出到文件: {output_filename}")