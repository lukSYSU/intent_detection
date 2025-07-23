import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import namedtuple
from FlagEmbedding import FlagReranker

transformers.logging.set_verbosity_error()

# 定义一个与 pymilvus.model.reranker.BGERerankFunction 兼容的返回对象
RerankResult = namedtuple('RerankResult', ['text', 'score', 'metadata'])


class Qwen3Reranker:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(self.device)
        
        # 获取特殊token的ID
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        
        # 设置前缀和后缀tokens
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.max_length = 8192
    
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output
    
    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
        
    def build_rerank_pairs(self, query, documents, instruction=None):
        pairs = [f"{self.prefix}{self.format_instruction(instruction, query, doc)}{self.suffix}" for doc in documents]
        return pairs
    
    def __call__(self, query, documents, top_k=None, instruction=None):
        """兼容BGERerankFunction的调用方式"""
        # 兼容两种输入：纯字符串列表或LangChain Document对象列表
        docs_to_process = []
        is_doc_objects = False
        if documents and hasattr(documents[0], 'page_content'):
            is_doc_objects = True
            docs_to_process = [doc.page_content for doc in documents]
        else:
            docs_to_process = documents

        # 格式化输入
        pairs = [self.format_instruction(instruction, query, doc) for doc in docs_to_process]
        
        # 处理输入
        inputs = self.process_inputs(pairs)
        
        # 计算分数
        scores = self.compute_logits(inputs)
        
        # 创建结果列表
        results = [(doc, score) for doc, score in zip(documents, scores)]
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        if top_k is not None:
            results = results[:top_k]

        # 格式化为预期的输出结构
        if is_doc_objects:
             # 如果输入是Document对象，保留完整的对象和元数据
            return [RerankResult(text=doc.page_content, score=score, metadata=doc.metadata) for doc, score in results]
        else:
            # 如果输入是字符串，metadata为None
            return [RerankResult(text=doc, score=score, metadata=None) for doc, score in results]


class StandaloneBGEReranker:
    """
    一个独立的 BGE Reranker 包装类，使用 FlagEmbedding 库。
    它模仿了原 pymilvus.BGERerankFunction 的行为，使其可以无缝替换。
    """
    def __init__(self, model_path, device='cuda:0'):
        print(f"Initializing StandaloneBGEReranker with model: {model_path}")
        # FlagEmbedding 的 BGEReranker 会自动使用GPU（如果可用）。
        # use_fp16=True 在CUDA上能提升性能。
        # use_fp16 = True if 'cuda' in str(device) else False
        use_fp16 = False
        self.model = FlagReranker(model_path, use_fp16=use_fp16)
        print("StandaloneBGEReranker initialized.")

    def __call__(self, query: str, documents, top_k: int):
        """
        :param query: 查询字符串。
        :param documents: 文档列表。可以是字符串列表，也可以是LangChain的Document对象。
        :param top_k: 返回前K个结果。
        :return: 一个RerankResult对象的列表。
        """
        # 兼容两种输入：纯字符串列表或LangChain Document对象列表
        docs_to_process = []
        is_doc_objects = False
        if documents and hasattr(documents[0], 'page_content'):
            is_doc_objects = True
            docs_to_process = [doc.page_content for doc in documents]
        else:
            docs_to_process = documents

        pairs = [[query, doc] for doc in docs_to_process]
        
        if not pairs:
            return []

        # 计算分数
        scores = self.model.compute_score(pairs)
        
        # 将原始文档/对象与分数结合
        if is_doc_objects:
            results_with_scores = list(zip(documents, scores))
        else:
            results_with_scores = list(zip(docs_to_process, scores))

        # 按分数降序排序
        sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
        
        # 取前 k 个结果
        top_results = sorted_results[:top_k]
        
        # 格式化为预期的输出结构
        if is_doc_objects:
             # 如果输入是Document对象，保留完整的对象和元数据
            return [RerankResult(text=doc.page_content, score=score, metadata=doc.metadata) for doc, score in top_results]
        else:
            # 如果输入是字符串，metadata为None
            return [RerankResult(text=doc, score=score, metadata=None) for doc, score in top_results]

