from pymilvus import MilvusClient
from pymilvus import model
client = MilvusClient("metadata.db")
if client.has_collection(collection_name="metadata_collection"):
    client.drop_collection(collection_name="metadata_collection")
client.create_collection(
    collection_name="metadata_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)


embedding_fn = model.DefaultEmbeddingFunction()

docs = ["N/A"]
vectors = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": vectors[i], "doc_name": "original"}
    for i in range(len(vectors))
]
print(vectors[0])
print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="metadata_collection", data=data)

print(res)
query_vectors = embedding_fn.encode_queries(
    ["什么是土布整经？"]
)
res = client.search(
    collection_name="metadata_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["doc_name"],  # specifies fields to be returned
)

print(res)