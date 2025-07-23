import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
# from pymilvus.model.reranker import BGERerankFunction
from vllm import LLM
import logging

logging.basicConfig(level=logging.WARNING)

# 配置参数
# EMBEDDING_MODEL_PATH = "./bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = "./Qwen3-Embedding-0.6B"
# BGE_RERANKER_MODEL_PATH = "./bge-reranker-v2-m3"
QWEN3_RERANKER_MODEL_PATH = "./Qwen3-Reranker-8B"  # 可以改为本地路径
MILVUS_URI = "http://localhost:19530"
# COLLECTION_NAME = "LangChainCollection"
COLLECTION_NAME = "qwen3_embedding"
DEVICE = "cuda:0"
VECTOR_FIELDS = ["dense", "sparse"]
SPARSE_OUTPUT_FIELD = "sparse"

# Reranker类型配置
RERANKER_TYPE = "qwen3"  # 可选: "bge" 或 "qwen3"


class Qwen3Reranker:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        
        # 使用 vLLM 初始化模型
        self.model = LLM(
            model=model_path,
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )
        
        # 设置模板
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"
    
    def format_pairs(self, instruction, query, documents):
        """格式化查询和文档对"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        queries = [
            self.query_template.format(prefix=self.prefix, instruction=instruction, query=query)
            for _ in documents
        ]
        documents_formatted = [
            self.document_template.format(doc=doc, suffix=self.suffix) 
            for doc in documents
        ]
        
        return queries, documents_formatted
    
    def rerank(self, query, documents, top_k=None, instruction=None):
        # 格式化查询和文档对
        queries, documents_formatted = self.format_pairs(instruction, query, documents)
        
        # 使用 vLLM 的 score API 计算分数
        outputs = self.model.score(queries, documents_formatted)
        
        # 提取分数
        scores = [output.outputs.score for output in outputs]
        
        # 创建结果列表
        results = [(doc, score) for doc, score in zip(documents, scores)]
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def __call__(self, query, documents, top_k=None, instruction=None):
        """兼容BGERerankFunction的调用方式"""
        results = self.rerank(query, documents, top_k, instruction)
        
        # 创建类似BGERerankFunction返回的对象
        class RerankResult:
            def __init__(self, text, score):
                self.text = text
                self.score = score
        
        return [RerankResult(doc, score) for doc, score in results]


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": DEVICE},
    )


def create_reranker():
    """根据配置创建相应的reranker"""
    if RERANKER_TYPE == "bge":
        return BGERerankFunction(
            model=BGE_RERANKER_MODEL_PATH,
            device=DEVICE,
        )
    elif RERANKER_TYPE == "qwen3":
        return Qwen3Reranker(
            model_path=QWEN3_RERANKER_MODEL_PATH,
            device=DEVICE,
        )
    else:
        raise ValueError(f"不支持的reranker类型: {RERANKER_TYPE}。支持的类型: 'bge', 'qwen3'")


def connect_to_existing_vectorstore():
    """连接到已存在的向量存储"""
    embedding = create_embedding_model()
    
    vectorstore = Milvus(
        embedding_function=embedding,
        builtin_function=BM25BuiltInFunction(output_field_names=SPARSE_OUTPUT_FIELD),
        vector_field=VECTOR_FIELDS,
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
    )
    
    return vectorstore


def search_and_rerank(vectorstore, query, k, top_k, verbose=True):
    """搜索和重排"""
    reranker = create_reranker()
    
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
    retrieved_docs = []
    if verbose:
        print("召回的问题：")
    for i, doc in enumerate(results, 1):
        retrieved_docs.append(doc.page_content)
        if verbose:
            print(f"{i}. {doc.page_content}")
            print("-" * 50)
    
    # 执行重排序
    if verbose:
        print("正在重排")
    reranked_results = reranker(
        query=query,
        documents=retrieved_docs,
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


def load_test_data(json_file_path):
    """加载测试数据"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_recall(vectorstore, test_data, k, top_k):
    """评估召回成功率"""
    total_queries = len(test_data)
    successful_recalls = 0
    
    print(f"开始评估，共{total_queries}个查询...")
    
    for i, item in enumerate(test_data, 1):
        adversarial_question = item['adversarial_question']
        original_question = item['question']
        
        print(f"\n处理第{i}/{total_queries}个查询...")
        
        # 执行搜索和重排序
        reranked_results = search_and_rerank(
            vectorstore=vectorstore,
            query=adversarial_question,
            k=k,
            top_k=top_k,
            verbose=False
        )
        
        # 检查是否召回了正确的原始问题
        recall_success = False
        for result in reranked_results:
            if result.text.strip() == original_question.strip():
                recall_success = True
                break
        
        if recall_success:
            successful_recalls += 1
            print(f"✓ 召回成功")
        else:
            print(f"✗ 召回失败")
            print(f"查询: {adversarial_question}")
            print(f"期望: {original_question}")
            if reranked_results:
                print(f"实际召回: {reranked_results[0].text}")
    
    recall_rate = successful_recalls / total_queries
    print(f"\n=== 评估结果 ===")
    print(f"总查询数: {total_queries}")
    print(f"成功召回: {successful_recalls}")
    print(f"召回成功率: {recall_rate:.2%}")
    
    return recall_rate, successful_recalls, total_queries


def main():
    """主函数 - 执行批量评估"""
    # 连接到已存在的向量存储
    vectorstore = connect_to_existing_vectorstore()
    
    # 加载测试数据
    json_file_path = "/home/liuzihao/intent_detection/panwei_questions.json"
    test_data = load_test_data(json_file_path)
    
    print(f"加载了{len(test_data)}条测试数据")
    
    # 执行评估
    recall_rate, successful_recalls, total_queries = evaluate_recall(
        vectorstore=vectorstore,
        test_data=test_data,
        k=5,
        top_k=1
    )
    
    return recall_rate, successful_recalls, total_queries


if __name__ == "__main__":
    results = main()
    
    


