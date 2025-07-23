import utils as libMgmt
import tqdm
import pandas as pd
from multiprocessing import Pool, Manager
import os

def generate_distillation_data(vectorstore_params, reranker_params, test_data_slice, k, gpu_id):
    """
    评估召回成功率, 针对单个进程的数据分片和GPU ID.
    
    Args:
        vectorstore_params (dict): 包含连接向量存储所需的所有参数的字典。
        reranker_params (dict): 包含创建reranker所需的所有参数的字典。
        test_data_slice (list): 当前进程要处理的测试数据分片。
        k (int): 召回文档的数量。
        top_k (int): 重排序后保留的顶部文档数量。
        gpu_id (int): 当前进程使用的GPU ID。
    
    Returns:
        pandas.DataFrame: 包含重排序结果的DataFrame。
    """
    
    # 在每个进程中初始化向量存储和Reranker
    # 这确保了每个进程都有自己的GPU上下文和模型实例
    print(f"进程 {os.getpid()} 在 GPU {gpu_id} 上初始化模型...")
    vectorstore = libMgmt.connect_to_existing_vectorstore(
        vectorstore_params['milvus_conn_args'],
        vectorstore_params['collection_name'],
        vectorstore_params['embed_model_name_or_path'],
        f"cuda:{gpu_id}" # 使用进程对应的GPU
    )
    reranker = libMgmt.create_reranker(
        reranker_params['reranker_type'],
        reranker_params['ranker_model_name_or_path'],
        f"cuda:{gpu_id}" # 使用进程对应的GPU
    )

    total_queries = len(test_data_slice)
    print(f"进程 {os.getpid()} 开始评估，共{total_queries}个查询...")
    
    dfs = []

    # 使用tqdm为每个进程提供独立的进度条
    for i, item in tqdm.tqdm(enumerate(test_data_slice, 1), desc=f"进程 {os.getpid()}"):
        query = item['adversarial_question']

        recall_docs = libMgmt.semantic_vector_recall(vectorstore, query, k=k, verbose=False)
        reranked_results = libMgmt.rerank_documents(reranker, query, recall_docs, top_k=k, verbose=False)
        rerank_pairs = reranker.build_rerank_pairs(query, recall_docs)
        scores = [rr.score for rr in reranked_results]

        retri_num = len(recall_docs)
        if retri_num != k:
            print(f"****reranker doc num is not equal:k={k}, retri_num={retri_num}")

        temp_df = pd.DataFrame({'query':[query]*retri_num, 'doc':recall_docs, 'pair': rerank_pairs, 'score': scores})
        dfs.append(temp_df)
        
    # 如果dfs为空（例如，test_data_slice为空），则返回一个空的DataFrame
    if dfs:
        df = pd.concat(dfs, axis=0)
    else:
        df = pd.DataFrame(columns=['query', 'doc', 'pair', 'score']) # 保持DataFrame结构一致
    return df


if __name__ == "__main__":
    # JSON文件路径，包含测试数据
    json_file_path = "/home/liuzihao/intent_detection/panwei_questions.json"

    # Milvus 连接参数
    milvus_conn_args={
        "uri": "http://localhost:19530",
        "user":'',
        "password":'',
        "db_name":'default'
    }
    collection_name = "qwen3_embedding"
    embed_model_name_or_path = "./Qwen3-Embedding-0.6B"
    
    # Reranker 类型和模型路径
    reranker_type = 'qwen3'
    ranker_model_name_or_path = "./Qwen3-Reranker-4B" 

    output_file_base_name = "reranker_output_multi_process_qwen_E06_R4"

    # 将向量存储和Reranker的参数打包成字典，以便传递给子进程
    vectorstore_params = {
        "milvus_conn_args": milvus_conn_args,
        "collection_name": collection_name,
        "embed_model_name_or_path": embed_model_name_or_path,
    }

    reranker_params = {
        "reranker_type": reranker_type,
        "ranker_model_name_or_path": ranker_model_name_or_path,
    }

    # 加载测试数据
    test_data = libMgmt.load_json_data(json_file_path)
    # test_data = test_data[:10] # 调试时可以使用小部分数据，实际运行时请注释掉此行
    
    print(f"加载了{len(test_data)}条测试数据")

    # 定义要使用的进程数量和可用GPU ID
    # 请根据你的实际CPU核心数和GPU数量进行调整
    num_processes = 8 
    # 假设你的GPU ID从0开始，例如有4个GPU就是 [0, 1, 2, 3]
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7] 

    # 确保进程数量不超过可用GPU数量
    if num_processes > len(available_gpus):
        num_processes = len(available_gpus)
        print(f"进程数量调整为可用的GPU数量: {num_processes}")
    
    # 如果可用GPU数量不足，确保至少有一个进程
    if num_processes == 0 and len(available_gpus) > 0:
        num_processes = 1
        print(f"可用GPU数量大于0但进程数为0，将进程数调整为1。")
    elif num_processes == 0:
        print("没有可用的GPU，无法启动任何进程。请检查 available_gpus 配置。")
        exit()


    # 将测试数据分片，每个分片分配给一个进程
    # `data_slices[i::num_processes]` 将数据均匀分配
    data_slices = [test_data[i::num_processes] for i in range(num_processes)]

    # 使用Manager来管理进程池，确保进程间可以正确通信
    with Manager() as manager:
        # 创建进程池
        with Pool(processes=num_processes) as pool:
            results = []
            for i in range(num_processes):
                # 为每个进程分配一个GPU ID和数据分片
                # `i % len(available_gpus)` 实现GPU的循环使用
                gpu_id = available_gpus[i % len(available_gpus)] 
                results.append(pool.apply_async(generate_distillation_data, 
                                                (vectorstore_params, reranker_params, data_slices[i], 5, gpu_id)))
            
            # 获取所有进程的结果，并显示总进度
            dfs_from_processes = []
            for res in tqdm.tqdm(results, desc="等待所有进程完成"):
                try:
                    df_slice = res.get() # 获取子进程返回的DataFrame
                    if not df_slice.empty: # 只添加非空的DataFrame
                        dfs_from_processes.append(df_slice)
                except Exception as e:
                    # 打印进程中发生的错误，以便调试
                    print(f"一个进程发生了错误: {e}")

    # 合并所有进程返回的DataFrame
    if dfs_from_processes:
        final_df = pd.concat(dfs_from_processes, axis=0)
        print(f"****distillation data info (merged):{final_df.shape}")
        # 将最终结果保存到新的CSV文件
        final_df.to_excel('./{output_file_name}.xlsx', index=False)
    else:
        print("没有生成任何蒸馏数据。请检查输入数据、GPU配置或进程中的错误。")

