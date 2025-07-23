# -*- coding: utf-8 -*-
"""
基于 FAST API 封装的检索后端；
 在终端中运行:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# 假设这些模块与 main.py 在同一目录下
import config as cfgMgmt
import utils as libMgmt

# --- 应用状态管理 ---
# 使用一个字典来存储在应用生命周期内加载的模型和连接
# 这样可以避免在每次API调用时都重新加载，极大提升性能
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 的生命周期管理器。
    在应用启动时执行 yield 之前的部分，在应用关闭时执行 yield 之后的部分。
    """
    # --- 应用启动时执行 ---
    print("应用启动... 正在加载模型和连接数据库...")
    
    # 连接到 Milvus
    try:
        print(f"尝试连接到 Milvus 向量存储: {cfgMgmt.MILVUS_URI}...")
        app_state["vectorstore"] = libMgmt.connect_to_existing_vectorstore(
            cfgMgmt.MILVUS_CONN_ARGS,
            cfgMgmt.COLLECTION_NAME,
            cfgMgmt.EMBEDDING_MODEL_PATH,
            cfgMgmt.DEVICE
        )
        print("Milvus 向量存储连接成功。")
    except Exception as e:
        print(f"致命错误: 连接 Milvus 向量存储失败: {e}", file=sys.stderr)
        app_state["vectorstore"] = None

    # 加载 Reranker 模型
    try:
        print(f"正在加载 Reranker 模型 ({cfgMgmt.RERANKER_MODEL_PATH})...")
        app_state["reranker"] = libMgmt.create_reranker(
            cfgMgmt.RERANKER_TYPE,
            cfgMgmt.RERANKER_MODEL_PATH,
            cfgMgmt.DEVICE
        )
        print("Reranker 模型加载成功。")
    except Exception as e:
        print(f"致命错误: 加载 Reranker 模型失败: {e}", file=sys.stderr)
        app_state["reranker"] = None
        
    # 加载 Embedding 模型
    try:
        print(f"正在加载 Embedding 模型 ({cfgMgmt.EMBEDDING_MODEL_PATH})...")
        app_state["embedding_model"] = libMgmt.create_embedding_model(
            cfgMgmt.EMBEDDING_MODEL_PATH, 
            cfgMgmt.DEVICE
        )
        print("Embedding 模型加载成功。")
    except Exception as e:
        print(f"致命错误: 加载 Embedding 模型失败: {e}", file=sys.stderr)
        app_state["embedding_model"] = None

    yield

    # --- 应用关闭时执行 ---
    print("应用关闭... 正在清理资源...")
    app_state.clear()
    print("资源已清理。")

# --- FastAPI 应用实例 ---
# 使用 lifespan 管理器来加载模型
app = FastAPI(
    title="信息检索 API",
    description="一个用于文档检索和上传的 API 服务。",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic 数据模型定义 ---
# 使用 Pydantic 模型可以获得类型提示、数据校验和自动生成API文档的好处

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的检索查询文本。")
    top_k: int = Field(default=3, ge=1, le=20, description="需要返回的重排后文档数量。")

class SearchResult(BaseModel):
    score: float = Field(..., description="文档与查询的相关性分数。")
    text: str = Field(..., description="文档的内容。")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="文档的元数据。")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="检索和重排后的结果列表。")

class UploadResponse(BaseModel):
    message: str
    filename: str
    lines_processed: int

# --- API 端点 (Endpoints) ---

@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_documents(request: SearchRequest):
    """
    接收一个查询，从向量库中检索相关文档，经过重排后返回 Top-K 结果。
    """
    if not app_state.get("vectorstore") or not app_state.get("reranker"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    print(f"\n接收到查询: '{request.query}', 正在从 Milvus 检索...")
    try:
        # 召回阶段
        recall_docs = libMgmt.semantic_vector_recall(app_state["vectorstore"], request.query, k=20) # 召回更多以供精排
        
        # 精排阶段
        rerank_results = libMgmt.rerank_documents(app_state["reranker"], request.query, recall_docs, top_k=request.top_k)

        if not rerank_results:
            return SearchResponse(results=[])

        # 格式化为 Pydantic 模型
        response_results = [
            SearchResult(score=res.score, text=res.text, metadata=res.metadata)
            for res in rerank_results
        ]
        
        return SearchResponse(results=response_results)

    except Exception as e:
        print(f"检索过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


@app.post("/upload", response_model=UploadResponse, tags=["Data Management"])
async def upload_document(file: UploadFile = File(..., description="要上传和向量化的 .txt 文件。")):
    """
    上传一个 .txt 文件，将其内容按行分割，并存入 Milvus 向量库。
    """
    if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="文件格式错误，请上传 .txt 文件。")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        
        docs_to_add = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                docs_to_add.append(Document(page_content=line, metadata={"source": file.filename, "line_number": i + 1}))

        if not docs_to_add:
            raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的非空行。")

        # 将文档添加到 Milvus
        app_state["vectorstore"].add_documents(
            documents=docs_to_add,
            embedding=app_state["embedding_model"],
        )

        print(f"文件 '{file.filename}' 已上传并成功添加到 Milvus 向量库。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            lines_processed=len(docs_to_add)
        )
    except Exception as e:
        print(f"文件上传失败: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 打印配置信息以供检查
    print("--- 应用配置信息 ---")
    print(f"* Milvus URI: {cfgMgmt.MILVUS_URI}")
    print(f"* Collection Name: {cfgMgmt.COLLECTION_NAME}")
    print(f"* Embedding Model: {cfgMgmt.EMBEDDING_MODEL_PATH}")
    print(f"* Reranker Model: {cfgMgmt.RERANKER_MODEL_PATH}")
    print(f"* Device: {cfgMgmt.DEVICE}")
    print("----------------------")
    print("\n启动 FastAPI 服务...")
    print("访问 http://127.0.0.1:8000/docs 可查看自动生成的交互式API文档。")
    
    # 使用 uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)

