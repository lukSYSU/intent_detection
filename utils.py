import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus.model.reranker import BGERerankFunction

from reranker import Qwen3Reranker

def create_embedding_model(model_name_or_path, device='cuda:0'):
    return HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={"device": device},
    )


def create_reranker(reranker_type, ranker_model_name_or_path, device='cuda:0'):
    """根据配置创建相应的reranker"""
    # BGERerankFunction 是 pymilvus 的一部分
    if reranker_type == "bge":
        return BGERerankFunction(
            model=ranker_model_name_or_path,
            device=device,
        )
    
    elif reranker_type == "qwen3":
        return Qwen3Reranker(
            model_path=ranker_model_name_or_path,
            device=device,
        )
    
    else:
        raise ValueError(f"不支持的reranker类型: {reranker_type}。支持的类型: 'bge', 'qwen3'")


def connect_to_existing_vectorstore(milvus_conn_args, collection_name, embed_model_name_or_path, device):
    """连接到已存在的向量存储"""
    embedding = create_embedding_model(embed_model_name_or_path, device)
    
    vectorstore = Milvus(
        embedding_function=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
        vector_field=["dense", "sparse"],
        connection_args=milvus_conn_args,
        collection_name=collection_name,
    )
    
    return vectorstore


def semantic_vector_recall(vectorstore, query, k, verbose=False):
    """向量库召回，混合检索"""   
    if verbose:
        print(f"问题：{query}")
    
    # 执行相似性搜索
    if verbose:
        print("正在检索")
    results = vectorstore.similarity_search(
        query=query,
        k=k,
        ranker_type="rrf",
        ranker_params={"k": 100},
    )
    
    # 提取文档内容
    recall_docs = []
    if verbose:
        print("召回的问题：")
    for i, doc in enumerate(results, 1):
        recall_docs.append(doc.page_content)
        if verbose:
            print(f"{i}. {doc.page_content}")
            print("-" * 50)
    
    return recall_docs


def rerank_documents(reranker, query, docs, top_k, verbose=False):    
    if verbose:
        print(f"问题：{query}")
    
    # 执行重排序
    if verbose:
        print("正在重排")
    reranked_results = reranker(
        query=query,
        documents=docs,
        top_k=top_k,
    )
    
    # 显示重排序结果
    if verbose:
        print("重排后的结果：")
        for i, result in enumerate(reranked_results, 1):
            print(f"{i}. 相似度: {result.score}")
            print(f"   问题: {result.text}")
            print("-" * 50)
    
    return reranked_results


def load_json_data(json_file_path):
    """加载测试数据"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

    
    


