import gradio as gr
import os
import sys
import json
import torch
import transformers

import config as cfgMgmt
import utils as libMgmt



# --- 初始化 Milvus 相关组件 ---
# 这些对象应该只被创建一次
try:
    print(f"尝试连接到 Milvus 向量存储: {cfgMgmt.MILVUS_URI}...")
    vectorstore_instance = libMgmt.connect_to_existing_vectorstore(cfgMgmt.MILVUS_CONN_ARGS, 
                                                                   cfgMgmt.COLLECTION_NAME,
                                                                   cfgMgmt.EMBEDDING_MODEL_PATH, 
                                                                   cfgMgmt.DEVICE)
    print("Milvus 向量存储连接成功。")
except Exception as e:
    print(f"连接 Milvus 向量存储失败: {e}")
    print("请确保 Milvus 服务正在运行 (例如，通过 docker-compose)。")
    vectorstore_instance = None # 如果连接失败，设置为 None，以便在函数中处理

# 加载 reranker 模型，只加载一次
reranker_instance = None
try:
    print(f"正在加载 Reranker 模型 ({cfgMgmt.RERANKER_MODEL_PATH})...")
    reranker_instance = libMgmt.create_reranker(cfgMgmt.RERANKER_TYPE, 
                                                cfgMgmt.RERANKER_MODEL_PATH, 
                                                cfgMgmt.DEVICE)
    print("Reranker 模型加载成功。")
except Exception as e:
    print(f"加载 Reranker 模型失败: {e}")
    print("请确保模型文件已下载并放置在正确路径，且DEVICE配置正确。")


# --- 信息检索和文档上传逻辑 ---

def retrieve_top_k_documents(query: str, top_k: int = 3):
    """
    使用 Milvus 检索文档并进行重排。
    """
    if not vectorstore_instance or not reranker_instance:
        return ["系统未准备好：Milvus 数据库或 Reranker 模型未成功加载。请检查控制台错误信息。"]

    if not query:
        return ["请输入查询文本。"]

    print(f"\n接收到查询: '{query}', 正在从 Milvus 检索 Top {top_k} 文档...")
    try:
        recall_docs = libMgmt.semantic_vector_recall(vectorstore_instance, query, k=5)
        rerank_results = libMgmt.rerank_documents(reranker_instance, query, recall_docs, top_k=top_k)

        if not rerank_results:
            return [f"未找到与 '{query}' 相关的文档。"]

        formatted_results = []
        for i, result in enumerate(rerank_results):
            formatted_results.append(
                f"--- Top {i+1} --- \n"
                f"相似度: {result.score:.8f}\n"
                f"内容: {result.text}" # 限制显示长度
            )
        
        return "\n\n".join(formatted_results)

    except Exception as e:
        return [f"检索过程中发生错误: {e}"]


def upload_and_store_document(file_obj):
    """
    处理上传的文件，并将其存储到 Milvus 向量库。
    """
    if not vectorstore_instance:
        return "系统未准备好：Milvus 数据库未成功加载。请检查控制台错误信息。"

    if file_obj is None:
        return "请选择一个文件进行上传。"

    file_path = file_obj.name
    file_name = os.path.basename(file_path)

    try:
        docs_to_add = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f): 
                line = line.strip() 
                if line: # 只添加非空行
                    from langchain_core.documents import Document
                    docs_to_add.append(Document(page_content=line, metadata={"source": file_name, "line_number": line_num + 1}))

        if not docs_to_add:
            return f"文件 '{file_name}' 中没有可存储的非空行。"

        # 将文档添加到 Milvus
        embedding_model = libMgmt.create_embedding_model(cfgMgmt.EMBEDDING_MODEL_PATH, cfgMgmt.DEVICE) 
        
        vectorstore_instance.add_documents(
            documents=docs_to_add,
            embedding=embedding_model, # 需要传递 embedding 函数
        )

        print(f"文件 '{file_name}' 已上传并成功添加到 Milvus 向量库。")
        return f"文件 '{file_name}' 上传成功！"
    except Exception as e:
        print(f"文件上传失败: {e}")
        return f"文件上传失败: {e}"

# --- Gradio 界面定义 ---

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 基于 Milvus 的信息检索应用

        你可以输入文本查询来检索 Milvus 向量库中的文档，或上传新的文档以扩充知识库。
        确保 Milvus 服务和所有必要的模型已正确配置和运行。
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="输入检索查询 (Query)", placeholder="例如：苹果公司的创始人是谁？", lines=2)
            retrieve_button = gr.Button("检索 Top 3 文档")
        with gr.Column(scale=1):
            file_upload = gr.File(label="上传新文档 (.txt)", file_count="single", type="filepath")
            upload_button = gr.Button("上传并存储到 Milvus")

    output_text = gr.Textbox(label="检索结果", lines=10, interactive=False)
    
    # 提示当前 Milvus 状态，实际文档数量需要通过 Milvus API 查询
    # gr.Markdown(
    #     f"""
    #     **Milvus 配置信息：**
    #     * URI: `{cfgMgmt.MILVUS_URI}`
    #     * Collection Name: `{cfgMgmt.COLLECTION_NAME}`
    #     * Embedding Model Path: `{cfgMgmt.EMBEDDING_MODEL_PATH}`
    #     * Reranker Model Path: `{cfgMgmt.RERANKER_MODEL_PATH}`
    #     * Device: `{cfgMgmt.DEVICE}`
    #     """
    # )

    # 定义按钮点击事件
    retrieve_button.click(
        fn=lambda q: retrieve_top_k_documents(q, top_k=10), # 强制 top_k 为 3
        inputs=query_input,
        outputs=output_text
    )

    upload_button.click(
        fn=upload_and_store_document,
        inputs=file_upload,
        outputs=output_text # 将上传结果显示在输出框
    )

# 启动 Gradio 应用
if __name__ == "__main__":
    print(f"""
        **Milvus 配置信息：**
        * URI: `{cfgMgmt.MILVUS_URI}`
        * Collection Name: `{cfgMgmt.COLLECTION_NAME}`
        * Embedding Model Path: `{cfgMgmt.EMBEDDING_MODEL_PATH}`
        * Reranker Model Path: `{cfgMgmt.RERANKER_MODEL_PATH}`
        * Device: `{cfgMgmt.DEVICE}`
        """
    )

    demo.launch()