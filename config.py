model_base_dir = '/home/tanxiaobing/Models'

# 配置参数
# EMBEDDING_MODEL_PATH = "./bge-large-zh-v1.5"
EMBEDDING_MODEL_PATH = f"{model_base_dir}/Qwen3-Embedding-0.6B"

# RERANKER_TYPE = 'bge'
# RERANKER_MODEL_PATH = f"{model_base_dir}/bge-reranker-v2-m3"

RERANKER_TYPE = 'qwen3' 
# RERANKER_MODEL_PATH = f"{model_base_dir}/Qwen3-Reranker-4B" 
# RERANKER_MODEL_PATH = f"{model_base_dir}/Qwen3-Reranker-0.6B" 
RERANKER_MODEL_PATH = f"/home/tanxiaobing/IR_Rainbow/regression_model_output/checkpoint-300"

DEVICE = "cuda:6"

# Milvus配置
MILVUS_URI="http://localhost:19530"
MILVUS_CONN_ARGS={
    "uri": MILVUS_URI,
    "user":'',
    "password":'',
    "db_name":'default'
    }
# COLLECTION_NAME = "bge_embedding"
COLLECTION_NAME = "qwen3_embedding"


