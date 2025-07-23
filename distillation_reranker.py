import utils as libMgmt
import tqdm
import pandas as pd

def generate_distillation_data(vectorstore, reranker, test_data, k, top_k):
    """评估召回成功率"""
    total_queries = len(test_data)
    
    print(f"开始评估，共{total_queries}个查询...")
    
    dfs = []

    for i, item in tqdm.tqdm(enumerate(test_data, 1)):
        query = item['adversarial_question']
        ground_truth = item['question']
        
        print(f"\n处理第{i}/{total_queries}个查询...")
        
        recall_docs = libMgmt.semantic_vector_recall(vectorstore, query, k=k, verbose=False)
        reranked_results = libMgmt.rerank_documents(reranker, query, recall_docs, top_k=k, verbose=False)
        rerank_pairs = reranker.build_rerank_pairs(query, recall_docs)
        scores = [rr.score for rr in reranked_results]

        temp_df = pd.DataFrame({'pair':rerank_pairs, 'score':scores})

        dfs.append(temp_df)
        
    df = pd.concat(dfs, axis=0)
    return df


if __name__ == "__main__":
    json_file_path = "/home/liuzihao/intent_detection/panwei_questions.json"

    milvus_conn_args={
        "uri": "http://localhost:19530",
        "user":'',
        "password":'',
        "db_name":'default'
        }
    collection_name = "qwen3_embedding"
    embed_model_name_or_path = "./Qwen3-Embedding-0.6B"
    device = "cuda:3"

    reranker_type = 'qwen3'
    # ranker_model_name_or_path = "./Qwen3-Reranker-0.6B" 
    ranker_model_name_or_path = "./Qwen3-Reranker-4B" 

    # 连接到已存在的向量存储
    vectorstore = libMgmt.connect_to_existing_vectorstore(milvus_conn_args, collection_name, embed_model_name_or_path, device)
    reranker = libMgmt.create_reranker(reranker_type, ranker_model_name_or_path, device)
    
    # 加载测试数据
    test_data = libMgmt.load_json_data(json_file_path)
    # test_data = test_data[:10]
    
    print(f"加载了{len(test_data)}条测试数据")
    
    # 执行评估, 评估测试数据
    df = generate_distillation_data(
        vectorstore=vectorstore,
        reranker=reranker,
        test_data=test_data,
        k=5,
        top_k=1
    )
    
    print(f"****distillation data info:{df.shape}")
    df.to_csv('./reranker_output.csv', index=False)
    
    


