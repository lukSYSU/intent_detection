import utils as libMgmt
import tqdm
import pandas as pd

def evaluate_test_data(vectorstore, reranker, test_data, k, top_k):
    """评估召回成功率"""
    total_queries = len(test_data)
    successful_recalls = 0
    
    print(f"开始评估，共{total_queries}个查询...")
    
    output_dfs = []
    for i, item in tqdm.tqdm(enumerate(test_data, 1)):
        query = item['adversarial_question']
        ground_truth = item['question']
        
        print(f"\n处理第{i}/{total_queries}个查询...")
        
        recall_docs = libMgmt.semantic_vector_recall(vectorstore, query, k=k, verbose=False)
        reranked_results = libMgmt.rerank_documents(reranker, query, recall_docs, top_k=top_k, verbose=False)
        
        # 检查是否召回了正确的原始问题
        recall_success = False
        for result in reranked_results:
            if result.text.strip() == ground_truth.strip():
                recall_success = True
                break
        
        if recall_success:
            successful_recalls += 1
            print(f"✓ 召回成功")
        else:
            print(f"✗ 召回失败")
            print(f"查询: {query}")
            print(f"期望: {ground_truth}")
            if reranked_results:
                print(f"实际召回: {reranked_results[0].text}")
        
        tmp_df = pd.DataFrame({'query':[query], 'doc':[reranked_results[0].text], 'score':[reranked_results[0].score]})
        output_dfs.append(tmp_df)

    df = pd.concat(output_dfs, axis=0)

    recall_rate = successful_recalls / total_queries
    print(f"\n=== 评估结果 ===")
    print(f"总查询数: {total_queries}")
    print(f"成功召回: {successful_recalls}")
    print(f"召回成功率: {recall_rate:.2%}")
    

    return recall_rate, successful_recalls, total_queries, df


if __name__ == "__main__":
    json_file_path = "/home/liuzihao/intent_detection/panwei_questions.json"
    # json_file_path = "/home/liuzihao/intent_detection/saas_questions.json"
    model_base_dir = '/home/tanxiaobing/Models'

    milvus_conn_args={
        "uri": "http://localhost:19530",
        "user":'',
        "password":'',
        "db_name":'default'
        }
    collection_name = "qwen3_embedding"
    # collection_name = "qwen3_saas"
    embed_model_name_or_path = f"{model_base_dir}/Qwen3-Embedding-0.6B"
    device = "cuda:4"

    reranker_type = 'qwen3'
    # ranker_model_name_or_path = f"{model_base_dir}/Qwen3-Reranker-0.6B" 
    # ranker_model_name_or_path = f"{model_base_dir}/Qwen3-Reranker-4B" 
    ranker_model_name_or_path = "/home/tanxiaobing/IR_Rainbow/regression_model_output/checkpoint-300"

    output_file = "/home/tanxiaobing/IR_Rainbow/search_qwen_E06_R06O3.xlsx"

    # 连接到已存在的向量存储
    vectorstore = libMgmt.connect_to_existing_vectorstore(milvus_conn_args, collection_name, embed_model_name_or_path, device)
    reranker = libMgmt.create_reranker(reranker_type, ranker_model_name_or_path, device)
    
    # 加载测试数据
    test_data = libMgmt.load_json_data(json_file_path)
    # test_data = test_data[:100]
    
    print(f"加载了{len(test_data)}条测试数据")
    
    # 执行评估, 评估测试数据
    recall_rate, successful_recalls, total_queries, df = evaluate_test_data(
        vectorstore=vectorstore,
        reranker = reranker,
        test_data=test_data,
        k=5,
        top_k=1
    )

    df.to_excel(f"{output_file}", index=False)
    

