'''
使用ElasticSearch作为后端向量存储的数据库；
将文档存储到ES中；
'''

import json
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
import utils_es as libMgmt

def create_es_vectorstore_from_documents(embedding, data_file_path, doc_key, index_name, es_connection_args):
    """
    从JSON文件中的文档创建一个Elasticsearch向量存储。

    :param embedding: 用于生成文本嵌入的模型。
    :param data_file_path: 包含文档的JSON文件路径。
    :param doc_key: JSON对象中包含文本内容的键。
    :param index_name:要在Elasticsearch中创建或使用的索引名称。
    :param es_connection_args: 连接到Elasticsearch的参数字典。
    """
    print("--- 正在创建Elasticsearch向量存储 ---")
    print(f"数据文件: {data_file_path}")
    print(f"文档键: {doc_key}")
    print(f"ES索引名: {index_name}")
    print(f"ES连接参数: {es_connection_args}")

    # 1. 加载和解析数据文件
    try:
        with open(data_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 将JSON数据转换为LangChain的Document对象
        docs = [Document(page_content=doc[doc_key], metadata={"mermaid_txt": data_file_path}) for doc in data]
        print(f"成功加载并解析了 {len(docs)} 篇文档。")
    except Exception as e:
        print(f"错误：加载或解析数据文件失败: {e}")
        return None

    # 2. 使用 LangChain 的 ElasticsearchStore.from_documents 方法
    # 这个方法非常强大，会自动处理以下事情:
    # - 连接到Elasticsearch
    # - 如果索引不存在，则创建一个新的索引
    # - 自动创建合适的字段映射(mapping)，包括一个用于BM25的text字段和一个用于向量的dense_vector字段
    # - 将文档文本和其向量嵌入批量写入索引
    try:
        print("正在将文档写入Elasticsearch，这可能需要一些时间...")
        vectorstore = ElasticsearchStore.from_documents(
            documents=docs,
            embedding=embedding,
            index_name=index_name,
            **es_connection_args,
            # 如果索引已存在且你想清空它，可以用这个参数
            # 注意：这在 langchain_elasticsearch 的较新版本中可能需要通过 es_client 手动删除
            # 这里我们用一种更稳健的方式，在外部处理删除逻辑
        )
        print("向量存储创建和数据写入完成！")
        return vectorstore

    except Exception as e:
        print(f"错误：创建Elasticsearch向量存储失败: {e}")
        # 打印更详细的错误，例如连接问题
        print("请确保Elasticsearch服务正在运行，并且连接参数正确。")
        return None

if __name__ == "__main__":
    # --- 配置 ---
    # data_file_path = "./question_adversarial_saas.json"
    data_file_path = "./question_adversarial_panwei.json"
    doc_key = "question"

    model_base_dir = '/home/tanxiaobing/Models'
    embed_model_name_or_path = f"{model_base_dir}/Qwen3-Embedding-0.6B"
    device = 'cuda:7' # 根据你的环境调整


    # ES 中的 index 和 Milvus中的 collection 类似，都类似于关系型数据库中 表（Table）的概念；
    index_name = "qwen3_panwei_index" 
    drop_old = True # 如果为True，将在写入前删除旧索引

    # Elasticsearch 的连接参数
    es_connection_args = {
        "es_url": "http://localhost:9200"
        # 如果ES需要认证，可以添加 es_user 和 es_password
        # "es_user": "elastic",
        # "es_password": "your_password"
    }


    # --- 执行 ---
    print("开始执行数据存储流程...")
    
    # 创建嵌入模型
    embedding = libMgmt.create_embedding_model(embed_model_name_or_path, device)
    
    # 如果需要，先删除旧索引
    if drop_old:
        try:
            from elasticsearch import Elasticsearch
            es_client = Elasticsearch(hosts=[es_connection_args["es_url"]])
            if es_client.indices.exists(index=index_name):
                print(f"发现旧索引 '{index_name}'，正在删除...")
                es_client.indices.delete(index=index_name)
                print("旧索引删除成功。")
            es_client.close()
        except Exception as e:
            print(f"警告：删除旧索引时发生错误（可能是索引不存在），将继续执行。错误信息: {e}")

    # 创建新的向量存储
    create_es_vectorstore_from_documents(
        embedding, 
        data_file_path, 
        doc_key, 
        index_name, 
        es_connection_args
    )
    
    print("\n数据存储流程执行完毕。")
