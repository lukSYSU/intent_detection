# -*- coding: utf-8 -*-
"""
api_client.py
这是一个用于调用信息检索 FastAPI 服务的 Python 客户端脚本。

如何运行:
1. 首先，确保你的 FastAPI 服务正在运行。在另一个终端中，运行:
   uvicorn main:app --host 127.0.0.1 --port 8000
2. 确保你已经安装了 requests 库:
   pip install requests
3. 运行此脚本:
   python api_client.py
"""

import requests
import json
import os

# --- 配置 ---
# 你的 FastAPI 服务器运行的地址
API_BASE_URL = "http://127.0.0.1:8005"

def call_search_api(query: str, top_k: int = 3):
    """
    调用 /search 接口进行文档检索。

    Args:
        query (str): 检索的查询语句。
        top_k (int): 希望返回的结果数量。

    Returns:
        dict: API 返回的 JSON 数据。
    """
    search_url = f"{API_BASE_URL}/search"
    payload = {"query": query, "top_k": top_k}
    
    print(f"\n--- 正在调用搜索接口 ---")
    print(f"URL: {search_url}")
    print(f"Payload: {payload}")

    try:
        # 发送 POST 请求，使用 json 参数来传递 JSON 数据
        response = requests.post(search_url, json=payload, timeout=30)
        
        # 检查响应状态码
        response.raise_for_status()  # 如果状态码不是 2xx，将抛出 HTTPError

        print("✅ 搜索成功！")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 搜索失败: {e}")
        # 尝试打印更详细的错误信息
        try:
            print("服务器返回的错误详情:", response.json())
        except (json.JSONDecodeError, AttributeError):
            print("无法解析服务器返回的错误信息。")
        return None

def call_rerank_api(query: str, documents: list, top_k: int = 3):
    """
    调用 /rerank 接口对文档进行重排。

    Args:
        query (str): 检索的查询语句。
        documents (list): 待重排的文档列表。
        top_k (int): 希望返回的结果数量。

    Returns:
        dict: API 返回的 JSON 数据。
    """
    rerank_url = f"{API_BASE_URL}/rerank"
    payload = {"query": query, "documents": documents, "top_k": top_k}
    
    print(f"\n--- 正在调用重排接口 ---")
    print(f"URL: {rerank_url}")
    # 使用json.dumps打印payload，以便更好地显示中文
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(rerank_url, json=payload, timeout=30)
        response.raise_for_status()
        print("✅ 重排成功！")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 重排失败: {e}")
        try:
            print("服务器返回的错误详情:", response.json())
        except (json.JSONDecodeError, AttributeError):
            print("无法解析服务器返回的错误信息。")
        return None

def call_upload_txt_api(file_path: str):
    """
    调用 /upload_txt 接口上传 .txt 文件。

    Args:
        file_path (str): 要上传的 .txt 文件的路径。
    
    Returns:
        dict: API 返回的 JSON 数据。
    """
    upload_url = f"{API_BASE_URL}/upload_txt"
    
    print(f"\n--- 正在调用 TXT 文件上传接口 ---")
    print(f"URL: {upload_url}")
    print(f"File: {file_path}")

    try:
        # 准备要上传的文件
        # 'file' 这个键名必须与 FastAPI @app.post("/upload") 中
        # File() 的参数名一致。在您的代码中是 `file`。
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/plain')}
            
            # 发送 POST 请求，使用 files 参数来上传文件
            response = requests.post(upload_url, files=files, timeout=60)
            
            # 检查响应状态码
            response.raise_for_status()

            print(f"✅ TXT 文件 '{file_path}' 上传成功！")
            return response.json()
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ TXT 文件上传失败: {e}")
        try:
            print("服务器返回的错误详情:", response.json())
        except (json.JSONDecodeError, AttributeError):
            print("无法解析服务器返回的错误信息。")
        return None

def call_upload_json_api(file_path: str):
    """
    调用 /upload_json 接口上传 .json 文件。

    Args:
        file_path (str): 要上传的 .json 文件的路径。
        
    Returns:
        dict: API 返回的 JSON 数据。
    """
    upload_url = f"{API_BASE_URL}/upload_json"

    print(f"\n--- 正在调用 JSON 文件上传接口 ---")
    print(f"URL: {upload_url}")
    print(f"File: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/json')}
            response = requests.post(upload_url, files=files)
            response.raise_for_status()
            print(f"✅ JSON 文件 '{file_path}' 上传成功！")
            return response.json()
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ JSON 文件上传失败: {e}")
        try:
            print("服务器返回的错误详情:", response.json())
        except (json.JSONDecodeError, AttributeError):
            print("无法解析服务器返回的错误信息。")
        return None

def create_dummy_files():
    """创建用于测试的临时文件"""
    # 创建 .txt 文件
    txt_file_path = "sample_data.txt"
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("第一行是关于Python编程的技巧。\n")
        f.write("第二行介绍了什么是向量数据库。\n")
        f.write("FastAPI是一个现代、快速的Web框架。\n")
    print(f"创建了测试文件: {txt_file_path}")

    # 创建 .json 文件
    json_file_path = "sample_data.json"
    # JSON文件的内容示例
    json_data = [
        {
            "text": "Mermaid是一种基于文本的图表绘制工具。",
            "metadata": {
                "source": "manual_doc_1.md",
                "mermaid_txt": "graph TD; A-->B;"
            }
        },
        {
            "text": "Elasticsearch不仅是搜索引擎，还能存储和分析向量数据。",
            "metadata": {
                "source": "tech_blog_post.html",
                "mermaid_txt": ""
            }
        }
    ]
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print(f"创建了测试文件: {json_file_path}")
    
    return txt_file_path, json_file_path

def cleanup_dummy_files(txt_path, json_path):
    """删除测试后创建的临时文件"""
    if os.path.exists(txt_path):
        os.remove(txt_path)
        print(f"\n删除了测试文件: {txt_path}")
    if os.path.exists(json_path):
        os.remove(json_path)
        print(f"删除了测试文件: {json_path}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 创建测试文件
    sample_txt, sample_json = create_dummy_files()

    # 2. 测试上传 .txt 文件
    txt_upload_result = call_upload_txt_api(sample_txt)
    if txt_upload_result:
        print("服务器响应:", json.dumps(txt_upload_result, indent=2, ensure_ascii=False))

    # 3. 测试上传 .json 文件
    json_upload_result = call_upload_json_api(sample_json)
    if json_upload_result:
        print("服务器响应:", json.dumps(json_upload_result, indent=2, ensure_ascii=False))

    # 4. 测试搜索接口
    #    假设上传的文档已经被ES索引，可以进行搜索
    query = "什么是FastAPI？"
    search_results = call_search_api(query)
    if search_results:
        print("服务器响应:", json.dumps(search_results, indent=2, ensure_ascii=False))
        
    query_2 = "介绍一下图表工具"
    search_results_2 = call_search_api(query_2)
    if search_results_2:
        print("服务器响应:", json.dumps(search_results_2, indent=2, ensure_ascii=False))

    # 5. 测试重排接口
    rerank_query = "哪种工具适合画图？"
    docs_to_rerank = [
        "Elasticsearch是一个强大的搜索引擎。",
        "Mermaid是用于从文本生成图表的工具。",
        "Python是一种流行的编程语言。",
        "LangChain可以帮助构建LLM应用。",
        "Graphviz也是一个流行的图表绘制软件。"
    ]
    rerank_results = call_rerank_api(rerank_query, docs_to_rerank, top_k=2)
    if rerank_results:
        print("服务器响应:", json.dumps(rerank_results, indent=2, ensure_ascii=False))

    # 6. 清理测试文件
    cleanup_dummy_files(sample_txt, sample_json)
