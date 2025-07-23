# -*- coding: utf-8 -*-
"""
基于 FAST API 封装的检索后端；
 在终端中运行:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# 假设这些模块与 main.py 在同一目录下
import config_es as cfgMgmt
import utils_es as libMgmt

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
    
    # 连接到 ES
    try:
        print(f"尝试连接到 ES 向量存储: {cfgMgmt.ES_URI}...")
        app_state["vectorstore"] = libMgmt.connect_to_existing_vectorstore(
            cfgMgmt.ES_CONN_ARGS,
            cfgMgmt.INDEX_NAME,
            cfgMgmt.EMBEDDING_MODEL_PATH,
            cfgMgmt.DEVICE
        )
        print("ES 向量存储连接成功。")
    except Exception as e:
        print(f"致命错误: 连接 ES 向量存储失败: {e}", file=sys.stderr)
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

class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的检索查询文本。")
    documents: List[str] = Field(..., min_length=1, description="待重排的文档内容列表。")
    top_k: int = Field(default=3, ge=1, le=50, description="需要返回的重排后文档数量。")

class SearchResult(BaseModel):
    score: float = Field(..., description="文档与查询的相关性分数。")
    text: str = Field(..., description="文档的内容。")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="文档的元数据。")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="检索和重排后的结果列表。")

class UploadResponse(BaseModel):
    message: str
    filename: str
    items_processed: int # 字段名修改得更通用

# --- API 端点 (Endpoints) ---

@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_documents(request: SearchRequest):
    """
    接收一个查询，从向量库中检索相关文档，经过重排后返回 Top-K 结果。
    """
    if not app_state.get("vectorstore") or not app_state.get("reranker"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    print(f"\n接收到查询: '{request.query}', 正在从 ES 检索...")
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

@app.post("/rerank", response_model=SearchResponse, tags=["Reranking"])
async def rerank_only(request: RerankRequest):
    """
    接收一个查询和一组文档，仅使用 Reranker 模型进行重排，并返回 Top-K 结果。
    这个端点不涉及从向量库的召回。
    """
    if not app_state.get("reranker"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：Reranker 模型未成功加载。")

    print(f"\n接收到 rerank 请求: query='{request.query}', documents_count={len(request.documents)}")
    
    try:
        # 将输入的字符串列表转换为 LangChain Document 对象
        # rerank_documents 函数期望的是 Document 对象
        docs_to_rerank = [Document(page_content=doc) for doc in request.documents]

        # 调用重排函数
        rerank_results = libMgmt.rerank_documents(
            app_state["reranker"], 
            request.query, 
            docs_to_rerank, 
            top_k=request.top_k
        )

        if not rerank_results:
            return SearchResponse(results=[])

        # 格式化为 Pydantic 模型
        response_results = [
            SearchResult(score=res.score, text=res.text, metadata=res.metadata)
            for res in rerank_results
        ]
        
        return SearchResponse(results=response_results)

    except Exception as e:
        print(f"重排过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


@app.post("/upload_txt", response_model=UploadResponse, tags=["Data Management"])
async def upload_text_document(file: UploadFile = File(..., description="要上传和向量化的 .txt 文件。")):
    """
    上传一个 .txt 文件，将其内容按行分割，并存入 ES 向量库。
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

        # 将文档添加到 ES
        app_state["vectorstore"].add_documents(
            documents=docs_to_add,
            embedding=app_state["embedding_model"],
        )

        print(f"文件 '{file.filename}' 已上传并成功添加到 ES 向量库。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            items_processed=len(docs_to_add)
        )
    except Exception as e:
        print(f"文件上传失败: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")

@app.post("/upload_json", response_model=UploadResponse, tags=["Data Management"])
async def upload_json_document(file: UploadFile = File(..., description="包含文档列表的 .json 文件。")):
    """
    上传一个 .json 文件，其内容为一个对象列表，每个对象包含 'text' 和 'metadata' 字段。
    将每个对象的 'text' 作为文档内容存入 ES 向量库。
    """
    if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="文件格式错误，请上传 .json 文件。")

    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))

        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON 文件的顶层结构必须是一个列表 (list)。")

        docs_to_add = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"JSON 列表中的第 {i+1} 个元素不是一个字典。")
            
            text = item.get("text")
            metadata = item.get("metadata")

            if not text or not isinstance(text, str):
                raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'text' 字段缺失或非字符串。")
            
            if not metadata or not isinstance(metadata, dict):
                 raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'metadata' 字段缺失或非字典。")
            
            # 直接使用提供的 text 和 metadata 创建 LangChain Document
            docs_to_add.append(Document(page_content=text, metadata=metadata))

        if not docs_to_add:
            raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的数据项。")

        # 将文档批量添加到 ES
        app_state["vectorstore"].add_documents(
            documents=docs_to_add,
            embedding=app_state["embedding_model"],
        )

        print(f"JSON 文件 '{file.filename}' 已上传并成功添加到 ES 向量库。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            items_processed=len(docs_to_add)
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 格式，文件解析失败。")
    except Exception as e:
        # 重新抛出已定义的HTTP异常，以便客户端获取准确的错误信息
        if isinstance(e, HTTPException):
            raise e
        # 其他未知错误则记录日志并返回500
        print(f"JSON 文件处理过程中发生未知错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 打印配置信息以供检查
    print("--- 应用配置信息 ---")
    print(f"* ES URI: {cfgMgmt.ES_URI}")
    print(f"* Index Name: {cfgMgmt.INDEX_NAME}")
    print(f"* Embedding Model: {cfgMgmt.EMBEDDING_MODEL_PATH}")
    print(f"* Reranker Model: {cfgMgmt.RERANKER_MODEL_PATH}")
    print(f"* Device: {cfgMgmt.DEVICE}")
    print("----------------------")
    print("\n启动 FastAPI 服务...")
    print("访问 http://127.0.0.1:8000/docs 可查看自动生成的交互式API文档。")
    
    # 使用 uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8005)

