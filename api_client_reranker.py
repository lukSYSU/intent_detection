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


if __name__ == "__main__":
    # 测试重排接口
    rerank_query = "表示肯定的、正面的、积极的"
    docs_to_rerank = [
        "查杀成功",
        "查杀失败",
        "查杀不成功",
        "执行失败",
        "可以查杀",
        "不可以查杀",
        "不能查杀",
        "无膨胀情况",
        "存在膨胀情况",
    ]
    rerank_results = call_rerank_api(rerank_query, docs_to_rerank, top_k=9)
    if rerank_results:
        print("服务器响应:", json.dumps(rerank_results, indent=2, ensure_ascii=False))

