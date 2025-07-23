import utils_es as libMgmt
import tqdm
import pandas as pd
from multiprocessing import Pool, Manager
import os

def evaluate_test_data(vectorstore_params, reranker_params, test_data, k, top_k, gpu_id):
    # 在每个进程中初始化向量存储和Reranker
    # 这确保了每个进程都有自己的GPU上下文和模型实例
    print(f"进程 {os.getpid()} 在 GPU {gpu_id} 上初始化模型...")
    vectorstore = libMgmt.connect_to_existing_vectorstore(
        vectorstore_params['es_conn_args'],
        vectorstore_params['index_name'],
        vectorstore_params['embed_model_name_or_path'],
        f"cuda:{gpu_id}" # 使用进程对应的GPU
    )
    reranker = libMgmt.create_reranker(
        reranker_params['reranker_type'],
        reranker_params['ranker_model_name_or_path'],
        f"cuda:{gpu_id}" # 使用进程对应的GPU
    )

    total_queries = len(test_data)
    print(f"进程 {os.getpid()} 开始评估，共{total_queries}个查询...")
    
    print(f"开始评估，共{total_queries}个查询...")
    successful_recalls = 0
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
    print(f"\n===GPU:{gpu_id} 评估结果 ===")
    print(f"总查询数: {total_queries}")
    print(f"成功召回: {successful_recalls}")
    print(f"召回成功率: {recall_rate:.2%}")    

    return successful_recalls, total_queries, df


if __name__ == "__main__":
    local_model_hub = '/home/tanxiaobing/Models'

    json_file_path = "/home/tanxiaobing/IR_Rainbow/data/questions_adversarial_panwei.json"
    # json_file_path = "/home/tanxiaobing/IR_Rainbow/data/questions_adversarial_saas.json"

    index_name = "qwen3_panwei_index"
    
    # embedding
    embed_model_name_or_path = f"{local_model_hub}/Qwen3-Embedding-0.6B"
    # embed_model_name_or_path = f"{local_model_hub}/bge-large-zh-v1.5"

    
    # reranker_type = 'bge'
    # ranker_model_name_or_path = f"{local_model_hub}/bge-reranker-v2-m3"

    reranker_type = 'qwen3'
    # ranker_model_name_or_path = f"{local_model_hub}/Qwen3-Reranker-0.6B" 
    ranker_model_name_or_path = f"{local_model_hub}/Qwen3-Reranker-4B" 
    # ranker_model_name_or_path = "/home/tanxiaobing/IR_Rainbow/regression_model_output/checkpoint-300"
    # ranker_model_name_or_path = "/home/tanxiaobing/IR_Rainbow/regression_model_output/checkpoint-50"

    output_file = "./output_search/search_qwen3_.xlsx"

    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7] 
    # available_gpus = [0,4, 5, 6, 7] 

    # Elasticsearch 的连接参数
    es_conn_args = {
        "es_url": "http://localhost:9200"
        # 如果ES需要认证，可以添加 es_user 和 es_password
        # "es_user": "elastic",
        # "es_password": "your_password"
    }
    
    # 将向量存储和Reranker的参数打包成字典，以便传递给子进程
    vectorstore_params = {
        "es_conn_args": es_conn_args,
        "index_name": index_name,
        "embed_model_name_or_path": embed_model_name_or_path,
    }

    reranker_params = {
        "reranker_type": reranker_type,
        "ranker_model_name_or_path": ranker_model_name_or_path,
    }

    # 加载测试数据
    test_data = libMgmt.load_json_data(json_file_path)
    # test_data = test_data[:5]
    
    print(f"加载了{len(test_data)}条测试数据")
    
    num_processes = len(available_gpus)
    data_slices = [test_data[i::num_processes] for i in range(num_processes)]

    total_success = 0
    total_queries = 0
    # 使用Manager来管理进程池，确保进程间可以正确通信
    with Manager() as manager:
        # 创建进程池
        with Pool(processes=num_processes) as pool:
            results = []
            for i in range(num_processes):
                gpu_id = available_gpus[i % len(available_gpus)] 
                results.append(pool.apply_async(evaluate_test_data, 
                                                (vectorstore_params, reranker_params, data_slices[i], 5, 1, gpu_id)))
            
            # 获取所有进程的结果，并显示总进度
            dfs_from_processes = []
            for res in tqdm.tqdm(results, desc="等待所有进程完成"):
                try:
                    success_cnt, query_cnt, df_slice = res.get() # 获取子进程返回的DataFrame
                    if not df_slice.empty: # 只添加非空的DataFrame
                        dfs_from_processes.append(df_slice)
                    total_success += success_cnt
                    total_queries += query_cnt
                except Exception as e:
                    # 打印进程中发生的错误，以便调试
                    print(f"一个进程发生了错误: {e}")

    # 合并所有进程返回的DataFrame
    if dfs_from_processes:
        final_df = pd.concat(dfs_from_processes, axis=0)
        print(f"****distillation data info (merged):{final_df.shape}")
        # 将最终结果保存到新的CSV文件
        final_df.to_excel(f'{output_file}', index=False)
    else:
        print("没有生成任何检索结果。请检查输入数据、GPU配置或进程中的错误。")

    recall_rate = total_success / total_queries
    print(f"\n=== Main 评估结果 ===")
    print(f"总查询数: {total_queries}")
    print(f"成功召回: {total_success}")
    print(f"召回成功率: {recall_rate:.2%}")  


