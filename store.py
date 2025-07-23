from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
import json
import utils as libMgmt
# 导入 pymilvus 的 connections 模块
from pymilvus import connections

def create_vectorstore_from_documents(embedding, data_file_path, doc_key, collection_name, milvus_conn_args, drop_old=False):
    """从文档创建向量存储"""

    print(f"正在创建向量存储...")
    print(f"data_file_path:{data_file_path}, doc_key:{doc_key}, collection_name={collection_name}, drop_old:{drop_old}")
    print(f"milvus_args:{milvus_conn_args}")
    
    # 加载数据
    with open(data_file_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    docs = [Document(page_content=doc[doc_key]) for doc in docs]
    print(f"*** docs num:{len(docs)}")

    # 创建向量存储
    # 这将在内部使用默认别名 'default' 创建一个连接
    vectorstore = Milvus.from_documents(
        collection_name=collection_name,
        documents=docs,
        embedding=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"), 
        vector_field=["dense", "sparse"],
        connection_args=milvus_conn_args,
        consistency_level="Strong",
        drop_old=drop_old,
    )

    print("向量存储创建完成！")
    
    return vectorstore



if __name__ == "__main__":    
    data_file_path = "./questions_adversarial_panwei.json"
    # data_file_path = "./questions_adversarial_saas.json"
    model_base_dir = '/home/tanxiaobing/Models'
    
    doc_key = "question"
    collection_name = "qwen3_panwei"
    # collection_name = "qwen3_saas"
    # collection_name = "bge_panwei"
    # collection_name = "bge_saas"
    drop_old = True


    # embed_model_name_or_path = f"{model_base_dir}/bge-large-zh-v1.5"
    embed_model_name_or_path = f"{model_base_dir}/Qwen3-Embedding-0.6B"
    device = 'cuda:7'

    milvus_conn_args={
        "uri": "http://localhost:19530",
        "user":'',
        "password":'',
        "db_name":'default'
        }
    
    embedding = libMgmt.create_embedding_model(embed_model_name_or_path, device)    
    create_vectorstore_from_documents(embedding, data_file_path, doc_key, collection_name, milvus_conn_args, drop_old)
    
    # ==================== 新增的修复代码 ====================
    # 在脚本完成所有工作后，显式地断开与 Milvus 的连接。
    # LangChain 在后台使用了默认的连接别名 'default'。
    try:
        print("\n正在断开 Milvus 连接...")
        connections.disconnect("default")
        print("连接已成功关闭。")
    except Exception as e:
        print(f"关闭 Milvus 连接时出错: {e}")
    # =======================================================
