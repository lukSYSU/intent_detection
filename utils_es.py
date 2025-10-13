''' 
utils.py , 重构为 Elasticsearch 版本
'''

import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore 
import sys
from langchain_core.documents import Document

# 你的 reranker 实现保持不变，因为它与向量数据库无关
from reranker import Qwen3Reranker, StandaloneBGEReranker

def create_embedding_model(model_name_or_path, device='cuda:0'):
    """创建嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={"device": device},
    )

def create_reranker(reranker_type, ranker_model_name_or_path, device='cuda:0'):
    """根据配置创建相应的reranker"""
    if reranker_type == "bge":
        return StandaloneBGEReranker(
            model_path=ranker_model_name_or_path, 
            device=device
        )
    
    elif reranker_type == "qwen3":
        return Qwen3Reranker(
            model_path=ranker_model_name_or_path,
            device=device,
        )
    
    else:
        raise ValueError(f"不支持的reranker类型: {reranker_type}。支持的类型: 'bge', 'qwen3'")


def connect_to_existing_vectorstore(es_connection_args, index_name, embedding_model, device):
    """
    连接到已存在的Elasticsearch向量存储（索引）。
    现在接收已加载的embedding模型实例而非模型路径
    """
    print(f"--- 正在连接到Elasticsearch索引: {index_name} ---")
    
    try:
        vectorstore = ElasticsearchStore(
            embedding=embedding_model,  # 直接使用传入的模型实例
            index_name=index_name,
            **es_connection_args
        )
        print("成功连接到Elasticsearch索引。")
        return vectorstore
    except Exception as e:
        print(f"错误：连接到Elasticsearch索引失败: {e}")
        return None


# def semantic_vector_recall(vectorstore: ElasticsearchStore, query: str, k: int, verbose: bool = False):
#     """
#     使用Elasticsearch内置的混合搜索（Hybrid Search）来召回文档。
#     该方法直接利用库的功能，在Elasticsearch服务端结合向量搜索和关键词搜索（BM25），
#     并使用倒数排名融合（RRF）来合并结果。
#     """
#     if verbose:
#         print(f"问题：{query}")
    
#     print("正在执行内置的混合检索 (BM25 + 向量)...")
    
#     # 直接调用内置的混合搜索方法
#     # 该方法会自动处理向量和关键词的组合以及RRF融合
#     # 注意：根据你的 langchian-elasticsearch 版本，方法名可能是 hybrid_search 或 similarity_search(..., search_type="hybrid")
#     # 我们这里使用 hybrid_search，因为它更直接地表达了意图。
#     try:
#         # 该方法会返回一个文档列表，已经按混合分数排序
#         final_docs = vectorstore.hybrid_search(query=query, k=k)
#     except AttributeError:
#         # 作为备选方案，一些版本的 retriever 可能这样调用
#         print("`hybrid_search` not found, trying `as_retriever` with hybrid search mode.")
#         retriever = vectorstore.as_retriever(search_type="hybrid", k=k)
#         final_docs = retriever.invoke(query)


#     if verbose:
#         print("混合检索后召回的文档：")
#         for i, doc in enumerate(final_docs, 1):
#             print(f"{i}. 内容: {doc.page_content}")
#             print(f"   来源: {doc.metadata.get('source', 'N/A')}")
#             # 注意：内置的 hybrid_search 可能不直接返回分数到 Document 对象中
#             print("-" * 50)
    
#     return final_docs

def semantic_vector_recall(vectorstore: ElasticsearchStore, query: str, k: int, verbose: bool = False):
    """
    手动执行混合搜索：分别获取向量和关键词（BM25）结果，
    然后使用倒数排名融合（RRF）将它们合并。
    这是一个在内置混合搜索不可用时的可靠替代方案。
    """
    if verbose:
        print(f"问题：{query}")
    
    print("正在执行手动混合检索 (BM25 + 向量)...")

    # 1. 执行向量搜索 (k-NN)
    vector_results = vectorstore.similarity_search_with_score(query=query, k=k)


    # 2. 使用底层的es_client执行关键词搜索 (BM25)
    keyword_results = []
    es_client = vectorstore.client
    # 针对 'text' 字段进行标准的 "match" 查询
    keyword_query = { "match": { "text": { "query": query } } }
    
    response = es_client.search(
        index=vectorstore._store.index,
        query=keyword_query,
        size=k
    )
    # 将返回的原始es结果解析为LangChain的Document对象
    for hit in response["hits"]["hits"]:
        doc = Document(
            page_content=hit["_source"].get("text", ""),
            metadata=hit["_source"].get("metadata", {})
        )
        score = hit["_score"]
        keyword_results.append((doc, score))


    # 3. 融合结果 (Reciprocal Rank Fusion - RRF)
    fused_scores = {}
    rrf_k_constant = 60  # RRF 的常用常量

    # 使用文档内容作为唯一标识符进行去重和融合
    def get_doc_id(doc):
        return doc.page_content

    # 处理向量搜索结果
    for rank, (doc, _) in enumerate(vector_results):
        doc_id = get_doc_id(doc)
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"doc": doc, "score": 0.0}
        fused_scores[doc_id]["score"] += 1.0 / (rrf_k_constant + rank + 1)

    # 处理关键词搜索结果
    for rank, (doc, _) in enumerate(keyword_results):
        doc_id = get_doc_id(doc)
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {"doc": doc, "score": 0.0}
        fused_scores[doc_id]["score"] += 1.0 / (rrf_k_constant + rank + 1)

    # 按 RRF 分数重新排序
    reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    # 提取前 k 个文档
    final_docs = [item["doc"] for item in reranked_results[:k]]

    if verbose:
        print("混合检索后召回的文档：")
        for i, doc in enumerate(final_docs, 1):
            print(f"{i}. 内容: {doc.page_content}")
            print(f"   来源: {doc.metadata.get('source', 'N/A')}")
            print("-" * 50)
    
    return final_docs


def rerank_documents(reranker, query, docs, top_k, verbose=False):    
    if verbose:
        print(f"问题：{query}")
    
    if verbose:
        print("正在重排")
    print("rerank_documents:")
    print("query:", query)
    print("docs:", docs)
    reranked_results = reranker(
        query=query,
        documents=docs,
        top_k=top_k,
    )
    
    if verbose:
        print("重排后的结果：")
        for i, result in enumerate(reranked_results, 1):
            # 假设 reranker 返回的对象有 score 和 text 属性
            print(f"{i}. 相似度: {result.score}")
            print(f"   内容: {result.text}")
            print("-" * 50)
    
    return reranked_results


def load_json_data(json_file_path):
    """加载测试数据（无变化）"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
