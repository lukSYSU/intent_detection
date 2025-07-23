from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
import json

# 配置参数
# EMBEDDING_MODEL_PATH = "./bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = "./Qwen3-Embedding-0.6B"
MILVUS_URI = "http://localhost:19530"
DEVICE = "cuda:1"
VECTOR_FIELDS = ["dense", "sparse"]
SPARSE_OUTPUT_FIELD = "sparse"
DATA_FILE_PATH = "panwei_questions.json"
org_quest_col_name = "question"


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": DEVICE},
    )


def create_vectorstore_from_documents(org_quest_col_name):
    """从文档创建向量存储"""
    embedding = create_embedding_model()
    
    # 加载数据
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    docs = [Document(page_content=doc[org_quest_col_name]) for doc in docs]
    print(f"*** docs num:{len(docs)}")

    # 创建向量存储
    vectorstore = Milvus.from_documents(
        collection_name="qwen3_embedding",
        documents=docs,
        embedding=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names=SPARSE_OUTPUT_FIELD), 
        vector_field=VECTOR_FIELDS,
        connection_args={"uri": MILVUS_URI},
        consistency_level="Strong",
        drop_old=True,
    )
    
    return vectorstore


def main():
    """主函数 - 创建向量存储"""
    print("正在创建向量存储...")
    vectorstore = create_vectorstore_from_documents(org_quest_col_name)
    print("向量存储创建完成！")
    return vectorstore


if __name__ == "__main__":
    vectorstore = main()


