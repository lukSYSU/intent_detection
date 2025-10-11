model_base_dir = '/home/tanxiaobing/Models'

# 配置参数
# EMBEDDING_MODEL_PATH = "./bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = f"{model_base_dir}/Qwen3-Embedding-4B"
# EMBEDDING_MODEL_PATH = f"{model_base_dir}/Qwen2.5-little-sparrow"


# RERANKER_TYPE = 'bge'
# RERANKER_MODEL_PATH = f"{model_base_dir}/bge-reranker-v2-m3"

RERANKER_TYPE = 'qwen3' 
# RERANKER_MODEL_PATH = f"{model_base_dir}/Qwen3-Reranker-4B" 
# RERANKER_MODEL_PATH = f"{model_base_dir}/Qwen3-Reranker-0.6B" 
RERANKER_MODEL_PATH = f"/home/tanxiaobing/IR_Rainbow/regression_model_output/checkpoint-300"

DEVICE = "cuda:0"

doc_key = "question"

# ES 中的 index 和 Milvus中的 collection 类似，都类似于关系型数据库中 表（Table）的概念；
INDEX_NAME = "qwen3_panwei_index" 
INDEX_NAME2 = "panwei_question_recall" 

# Elasticsearch 的连接参数
ES_URI = "http://localhost:1200"
ES_CONN_ARGS = {
    "es_url": ES_URI,
    # 如果ES需要认证，可以添加 es_user 和 es_password
    "es_user": "elastic",
    "es_password": "infini_rag_flow"
}