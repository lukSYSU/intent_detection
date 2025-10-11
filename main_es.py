# -*- coding: utf-8 -*-
"""
基于 FAST API 封装的检索后端；
 在终端中运行:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import hashlib
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
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

    # 加载 embedding 模型
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

    # 连接到 ES
    try:
        print(f"尝试连接到 ES 向量存储: {cfgMgmt.ES_URI}...")
        # 使用已加载的embedding模型
        app_state["vectorstore"] = libMgmt.connect_to_existing_vectorstore(
            cfgMgmt.ES_CONN_ARGS,
            cfgMgmt.INDEX_NAME,
            app_state["embedding_model"],  # 传入已加载的模型实例
            cfgMgmt.DEVICE
        )
        print("ES 向量存储连接成功。")
    except Exception as e:
        print(f"致命错误: 连接 ES 向量存储失败: {e}", file=sys.stderr)
        app_state["vectorstore"] = None

    # 连接到 ES2
    try:
        print(f"尝试连接到 ES 向量存储: {cfgMgmt.ES_URI}...")
        # 使用已加载的embedding模型
        app_state["vectorstore2"] = libMgmt.connect_to_existing_vectorstore(
            cfgMgmt.ES_CONN_ARGS,
            cfgMgmt.INDEX_NAME2,
            app_state["embedding_model"],  # 传入已加载的模型实例
            cfgMgmt.DEVICE
        )
        print("ES 向量存储连接成功。")
    except Exception as e:
        print(f"致命错误: 连接 ES 向量存储失败: {e}", file=sys.stderr)
        app_state["vectorstore2"] = None  # 修复此处的变量名错误


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

# --- 辅助函数 ---
def generate_doc_id(content: str, metadata: Optional[Dict] = None) -> str:
    """生成文档的唯一哈希ID"""
    data = content + (json.dumps(metadata, sort_keys=True) if metadata else "")
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

# --- Pydantic 数据模型定义 ---
# 使用 Pydantic 模型可以获得类型提示、数据校验和自动生成API文档的好处

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的检索查询文本。")
    top_k: int = Field(default=3, ge=1, le=20, description="需要返回的重排后文档数量。")

class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的检索查询文本。")
    documents: List[str] = Field(..., min_length=1, description="待重排的文档内容列表。")
    top_k: int = Field(default=3, ge=1, le=50, description="需要返回的重排后文档数量。")

class DeleteRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=1, description="要删除的文档ID列表。")

class UpdateRequest(BaseModel):
    doc_id: str = Field(..., description="要更新的文档ID。")
    text: str = Field(..., description="更新后的文档内容。")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="更新后的文档元数据。")

class SearchByIndexRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的检索查询文本。")
    index_name: str = Field(..., description="要查询的索引名称，支持qwen3_panwei_index和panwei_question_recall。")
    top_k: int = Field(default=3, ge=1, le=20, description="需要返回的重排后文档数量。")

class UploadToIndexRequest(BaseModel):
    file: UploadFile = File(..., description="要上传的文件（.txt或.json）。")
    index_name: str = Field(..., description="要上传到的索引名称，支持qwen3_panwei_index和panwei_question_recall。")

class DeleteFromIndexRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=1, description="要删除的文档ID列表。")
    index_name: str = Field(..., description="要删除文档的索引名称，支持qwen3_panwei_index和panwei_question_recall。")

class UpdateByIndexRequest(BaseModel):
    doc_id: str = Field(..., description="要更新的文档ID。")
    text: str = Field(..., description="更新后的文档内容。")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="更新后的文档元数据（仅qwen3_panwei_index有效）。")
    index_name: str = Field(..., description="目标索引名称，支持qwen3_panwei_index和panwei_question_recall。")

class SearchResult(BaseModel):
    doc_id: str = Field(..., description="文档的唯一ID。")
    score: float = Field(..., description="文档与查询的相关性分数。")
    text: str = Field(..., description="文档的内容。")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="文档的元数据。")

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="检索和重排后的结果列表。")

# 新增：简化的搜索结果模型（用于panwei_question_recall索引）
class SearchByIndexResult(BaseModel):
    doc_id: str = Field(..., description="文档的唯一ID。")
    text: str = Field(..., description="文档的内容。")
    score: float = Field(..., description="文档与查询的相关性分数（范围0-1）。")

class SearchByIndexResponse(BaseModel):
    results: List[SearchByIndexResult] = Field(..., description="检索和重排后的结果列表（仅包含doc_id和text）。")

class UploadResponse(BaseModel):
    message: str
    filename: str
    items_processed: int
    doc_ids: List[str] = Field(..., description="每个处理文档的唯一ID列表。")

class DeleteResponse(BaseModel):
    message: str
    deleted_count: int = Field(..., description="成功删除的文档数量。")

class UpdateResponse(BaseModel):
    message: str
    doc_id: str = Field(..., description="更新后的文档ID。")

# --- API 端点 (Endpoints) ---

@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_documents(request: SearchRequest):
    """
    接收一个查询，从默认向量库中检索相关文档，经过重排后返回 Top-K 结果。
    """
    if not app_state.get("vectorstore") or not app_state.get("reranker"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    print(f"\n接收到查询: '{request.query}', 正在从默认ES索引检索...")
    try:
        # 召回阶段
        recall_docs = libMgmt.semantic_vector_recall(app_state["vectorstore"], request.query, k=20) # 召回更多以供精排
        
        # 精排阶段
        rerank_results = libMgmt.rerank_documents(app_state["reranker"], request.query, recall_docs, top_k=request.top_k)

        if not rerank_results:
            return SearchResponse(results=[])

        # 格式化为 Pydantic 模型
        response_results = []
        for res in rerank_results:
            # 为每个结果生成唯一的 doc_id
            doc_id = generate_doc_id(res.text, res.metadata)
            response_results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=res.score,
                    text=res.text,
                    metadata=res.metadata
                )
            )
        
        return SearchResponse(results=response_results)

    except Exception as e:
        print(f"检索过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


# 新增：支持多索引的搜索接口
@app.post("/search_by_index", tags=["Retrieval"])
async def search_by_index(request: SearchByIndexRequest):
    """
    根据指定索引检索文档，两种索引均返回包含score的结果
    """
    # 状态检查：确保目标索引连接和reranker已加载
    if request.index_name == "qwen3_panwei_index":
        if not app_state.get("vectorstore") or not app_state.get("reranker"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore"]
    elif request.index_name == "panwei_question_recall":
        if not app_state.get("vectorstore2") or not app_state.get("reranker"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore2"]
    else:
        raise HTTPException(status_code=400, detail="无效的索引名称，仅支持qwen3_panwei_index和panwei_question_recall。")
    
    print(f"\n接收到查询: '{request.query}', 正在从索引 {request.index_name} 检索...")
    # 召回阶段
    print("召回开始")
    recall_docs = libMgmt.semantic_vector_recall(current_vectorstore, request.query, k=20)
    print("召回结束")

    # 精排阶段
    print("精排开始")
    rerank_results = libMgmt.rerank_documents(app_state["reranker"], request.query, recall_docs, top_k=request.top_k)
    print("精排结束")

    if not rerank_results:
        if request.index_name == "qwen3_panwei_index":
            return SearchResponse(results=[])
        else:
            return SearchByIndexResponse(results=[])
    
    # 根据索引类型格式化返回结果
    if request.index_name == "qwen3_panwei_index":
        # 返回完整结构
        response_results = []
        for res in rerank_results:
            doc_id = generate_doc_id(res.text, res.metadata)
            response_results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=res.score,
                    text=res.text,
                    metadata=res.metadata
                )
            )
        return SearchResponse(results=response_results)
    else:
        # 返回简化结构
        response_results = []
        for res in rerank_results:
            doc_id = generate_doc_id(res.text, res.metadata)
            response_results.append(
                SearchByIndexResult(
                    doc_id=doc_id,
                    score=res.score,
                    text=res.text
                )
            )
        return SearchByIndexResponse(results=response_results)


    

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
        response_results = []
        for res in rerank_results:
            doc_id = generate_doc_id(res.text, res.metadata)
            response_results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=res.score,
                    text=res.text,
                    metadata=res.metadata
                )
            )
        
        return SearchResponse(results=response_results)

    except Exception as e:
        print(f"重排过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


@app.post("/upload_txt", response_model=UploadResponse, tags=["Data Management"])
async def upload_text_document(file: UploadFile = File(..., description="要上传和向量化的 .txt 文件。")):
    """
    上传一个 .txt 文件到默认索引，将其内容按行分割，并存入 ES 向量库。
    """
    if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="文件格式错误，请上传 .txt 文件。")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        
        docs_to_add = []
        doc_ids = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                metadata = {"source": file.filename, "line_number": i + 1}
                doc_id = generate_doc_id(line, metadata)
                docs_to_add.append(Document(page_content=line, metadata=metadata))
                doc_ids.append(doc_id)

        if not docs_to_add:
            raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的非空行。")

        # 将文档添加到默认ES索引
        app_state["vectorstore"].add_documents(
            documents=docs_to_add,
            embedding=app_state["embedding_model"],
            ids=doc_ids
        )

        print(f"文件 '{file.filename}' 已上传并成功添加到默认ES向量库。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            items_processed=len(docs_to_add),
            doc_ids=doc_ids
        )
    except Exception as e:
        print(f"文件上传失败: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")

@app.post("/upload_json", response_model=UploadResponse, tags=["Data Management"])
async def upload_json_document(file: UploadFile = File(..., description="包含文档列表的 .json 文件。")):
    """
    上传一个 .json 文件到默认索引，其内容为一个对象列表，每个对象包含 'text' 和 'metadata' 字段。
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
        doc_ids = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"JSON 列表中的第 {i+1} 个元素不是一个字典。")
            
            text = item.get("text")
            metadata = item.get("metadata")
            doc_id = item.get("doc_id")

            if not text or not isinstance(text, str):
                raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'text' 字段缺失或非字符串。")
            
            if not metadata or not isinstance(metadata, dict):
                 raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'metadata' 字段缺失或非字典。")
            
            if not doc_id:
                doc_id = generate_doc_id(text, metadata)
            
            docs_to_add.append(Document(page_content=text, metadata=metadata))
            doc_ids.append(doc_id)

        if not docs_to_add:
            raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的数据项。")

        # 将文档批量添加到默认ES索引
        app_state["vectorstore"].add_documents(
            documents=docs_to_add,
            embedding=app_state["embedding_model"],
            ids=doc_ids
        )

        print(f"JSON 文件 '{file.filename}' 已上传并成功添加到默认ES向量库。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            items_processed=len(docs_to_add),
            doc_ids=doc_ids
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 格式，文件解析失败。")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"JSON 文件处理过程中发生未知错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")



# 新增：支持多索引的文件上传接口
@app.post("/upload_to_index", response_model=UploadResponse, tags=["Data Management"])
async def upload_to_index(
    file: UploadFile = File(..., description="要上传的文件（.txt 或 .json）。"),
    index_name: str = Form(..., description="目标索引名称。")
):
    """
    上传文件到指定的索引。
    - 支持.txt和.json格式
    - 根据文件类型自动处理内容
    """
    # 状态检查
    if index_name == "qwen3_panwei_index":
        if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore"]
    elif index_name == "panwei_question_recall":
        if not app_state.get("vectorstore2") or not app_state.get("embedding_model"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore2"]
    else:
        raise HTTPException(status_code=400, detail="无效的索引名称，仅支持qwen3_panwei_index和panwei_question_recall。")
    
    
    try:
        # 处理TXT文件
        if file.filename.endswith(".txt"):
            contents = await file.read()
            text = contents.decode("utf-8")
            
            docs_to_add = []
            doc_ids = []
            lines = text.splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    # 仅qwen3索引保留metadata
                    metadata = {"source": file.filename, "line_number": i + 1} if index_name == "qwen3_panwei_index" else {}
                    doc_id = generate_doc_id(line, metadata)
                    docs_to_add.append(Document(page_content=line, metadata=metadata))
                    doc_ids.append(doc_id)

            if not docs_to_add:
                raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的非空行。")

            current_vectorstore.add_documents(
                documents=docs_to_add,
                embedding=app_state["embedding_model"],
                ids=doc_ids
            )

        # 处理JSON文件
        elif file.filename.endswith(".json"):
            contents = await file.read()
            data = json.loads(contents.decode("utf-8"))
            if not isinstance(data, list):
                raise HTTPException(status_code=400, detail="JSON 文件的顶层结构必须是一个列表 (list)。")
            docs_to_add = []
            doc_ids = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise HTTPException(status_code=400, detail=f"JSON 列表中的第 {i+1} 个元素不是一个字典。")
                text = item.get("text")
                if not text or not isinstance(text, str):
                    raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'text' 字段缺失或非字符串。")
                # 仅qwen3索引保留metadata
                metadata = item.get("metadata") if index_name == "qwen3_panwei_index" else {}
                if index_name == "qwen3_panwei_index" and (not metadata or not isinstance(metadata, dict)):
                    raise HTTPException(status_code=400, detail=f"JSON 列表中第 {i+1} 个元素的 'metadata' 字段缺失或非字典。")
                doc_id = item.get("doc_id") or generate_doc_id(text, metadata)
                
                docs_to_add.append(Document(page_content=text, metadata=metadata))
                doc_ids.append(doc_id)
            if not docs_to_add:
                raise HTTPException(status_code=400, detail=f"文件 '{file.filename}' 中没有可处理的数据项。")
            current_vectorstore.add_documents(
                documents=docs_to_add,
                embedding=app_state["embedding_model"],
                ids=doc_ids
            )

        else:
            raise HTTPException(status_code=400, detail="文件格式错误，请上传 .txt 或 .json 文件。")

        print(f"文件 '{file.filename}' 已上传并成功添加到索引 {index_name}。")
        return UploadResponse(
            message=f"文件 '{file.filename}' 上传并处理成功。",
            filename=file.filename,
            items_processed=len(docs_to_add),
            doc_ids=doc_ids
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的 JSON 格式，文件解析失败。")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"文件上传到索引 {index_name} 失败: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {e}")



@app.post("/delete", response_model=DeleteResponse, tags=["Data Management"])
async def delete_documents(request: DeleteRequest):
    """
    根据文档ID列表删除默认索引中的文档。
    """
    if not app_state.get("vectorstore"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：向量存储未连接。")

    try:
        es_client = app_state["vectorstore"].client
        
        # 批量删除文档
        deleted_count = 0
        for doc_id in request.doc_ids:
            try:
                result = es_client.delete(
                    index=cfgMgmt.INDEX_NAME,
                    id=doc_id,
                    refresh=True
                )
                if result.get("result") == "deleted":
                    deleted_count += 1
            except Exception as e:
                print(f"删除文档 {doc_id} 时出错: {e}")
                continue
        
        return DeleteResponse(
            message=f"成功删除 {deleted_count} 个文档。",
            deleted_count=deleted_count
        )
    except Exception as e:
        print(f"删除文档过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")




# 新增：支持多索引的文档删除接口
@app.post("/delete_from_index", response_model=DeleteResponse, tags=["Data Management"])
async def delete_from_index(request: DeleteFromIndexRequest):
    """从指定索引中删除文档"""
    if request.index_name == "qwen3_panwei_index":
        if not app_state.get("vectorstore"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore"]
    elif request.index_name == "panwei_question_recall":
        if not app_state.get("vectorstore2"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore2"]
    else:
        raise HTTPException(status_code=400, detail="无效的索引名称，仅支持qwen3_panwei_index和panwei_question_recall。")
    
    
    try:
        # 批量删除文档
        es_client = current_vectorstore.client
        deleted_count = 0
        for doc_id in request.doc_ids:
            try:
                result = es_client.delete(
                    index=request.index_name,
                    id=doc_id,
                    refresh=True
                )
                if result.get("result") == "deleted":
                    deleted_count += 1
            except Exception as e:
                print(f"从索引 {request.index_name} 删除文档 {doc_id} 时出错: {e}")
                continue
        
        return DeleteResponse(
            message=f"从索引 {request.index_name} 成功删除 {deleted_count} 个文档。",
            deleted_count=deleted_count
        )
    except Exception as e:
        print(f"从索引 {request.index_name} 删除文档过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")

@app.post("/update", response_model=UpdateResponse, tags=["Data Management"])
async def update_document(request: UpdateRequest):
    """
    根据文档ID更新默认索引中的文档内容，并返回基于新内容生成的新哈希值。
    """
    if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
        raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")

    try:
        es_client = app_state["vectorstore"].client
        
        # 检查文档是否存在
        try:
            old_doc = es_client.get(
                index=cfgMgmt.INDEX_NAME,
                id=request.doc_id
            )
            if not old_doc.get("found"):
                raise HTTPException(status_code=404, detail=f"文档ID {request.doc_id} 不存在。")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"文档ID {request.doc_id} 不存在。")
        
        # 生成新的嵌入向量
        new_embedding = app_state["embedding_model"].embed_query(request.text)
        
        # 生成新的文档ID
        new_doc_id = generate_doc_id(request.text, request.metadata or {})
        
        # 准备更新文档
        document = {
            "text": request.text,
            "metadata": request.metadata or {},
            "vector": new_embedding
        }
        
        # 删除旧文档
        es_client.delete(
            index=cfgMgmt.INDEX_NAME,
            id=request.doc_id,
            refresh=True
        )
        
        # 插入新文档
        es_client.index(
            index=cfgMgmt.INDEX_NAME,
            id=new_doc_id,
            body=document,
            refresh=True
        )
        
        return UpdateResponse(
            message=f"文档更新成功。旧ID: {request.doc_id}, 新ID: {new_doc_id}",
            doc_id=new_doc_id
        )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"更新文档过程中发生错误: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"更新文档失败: {e}")

@app.post("/update_by_index", response_model=UpdateResponse, tags=["Data Management"])
async def update_document_by_index(request: UpdateByIndexRequest):
    """
    根据文档ID和指定索引更新ES中的文档内容，返回基于新内容生成的新哈希值。
    - qwen3_panwei_index: 支持完整元数据更新
    - panwei_question_recall: 自动忽略元数据，仅更新文本内容
    """
    if request.index_name == "qwen3_panwei_index":
        if not app_state.get("vectorstore") or not app_state.get("embedding_model"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore"]
    elif request.index_name == "panwei_question_recall":
        if not app_state.get("vectorstore2") or not app_state.get("embedding_model"):
            raise HTTPException(status_code=503, detail="服务暂时不可用：核心模型未成功加载。")
        current_vectorstore = app_state["vectorstore2"]
    else:
        raise HTTPException(status_code=400, detail="无效的索引名称，仅支持qwen3_panwei_index和panwei_question_recall。")
    
    
    try:
        es_client = current_vectorstore.client
        # 检查旧文档是否存在
        try:
            old_doc = es_client.get(index=request.index_name, id=request.doc_id)
            if not old_doc.get("found"):
                raise HTTPException(status_code=404, detail=f"索引 {request.index_name} 中文档ID {request.doc_id} 不存在。")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"索引 {request.index_name} 中文档ID {request.doc_id} 不存在。")
        
        # 处理元数据（根据索引类型适配）
        if request.index_name == "qwen3_panwei_index":
            # 强制校验元数据（若未传则用空对象，若传则校验格式）
            final_metadata = request.metadata or {}
            if not isinstance(final_metadata, dict):
                raise HTTPException(status_code=400, detail="qwen3_panwei_index 索引的 metadata 必须是字典格式。")
            # 可选：校验metadata必需字段（如source）
            if "source" not in final_metadata:
                final_metadata["source"] = "unknown"  # 或抛出错误，根据业务需求调整
        else:
            # panwei_question_recall 索引忽略元数据
            final_metadata = {}
        
        # 生成新嵌入向量和新文档ID
        new_embedding = app_state["embedding_model"].embed_query(request.text)
        new_doc_id = generate_doc_id(request.text, final_metadata)
        
        # 构建新文档（适配索引字段要求）
        new_document = {
            "text": request.text,
            "metadata": final_metadata,
            "vector": new_embedding
        }
        
        # 执行更新：删除旧文档 + 插入新文档
        es_client.delete(index=request.index_name, id=request.doc_id, refresh=True)
        es_client.index(index=request.index_name, id=new_doc_id, body=new_document, refresh=True)
        
        return UpdateResponse(
            message=f"索引 {request.index_name} 中文档更新成功。旧ID: {request.doc_id}, 新ID: {new_doc_id}",
            doc_id=new_doc_id
        )
    
    except HTTPException:
        raise  # 直接抛出已定义的HTTP异常
    except Exception as e:
        print(f"更新索引 {request.index_name} 中文档失败: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"更新文档失败: {e}")



# --- 主程序入口 ---
if __name__ == "__main__":
    # 打印配置信息以供检查
    print("--- 应用配置信息 ---")
    print(f"* ES URI: {cfgMgmt.ES_URI}")
    print(f"* 默认Index Name: {cfgMgmt.INDEX_NAME}")
    print(f"* 支持索引: qwen3_panwei_index, panwei_question_recall")
    print(f"* Embedding Model: {cfgMgmt.EMBEDDING_MODEL_PATH}")
    print(f"* Reranker Model: {cfgMgmt.RERANKER_MODEL_PATH}")
    print(f"* Device: {cfgMgmt.DEVICE}")
    print("----------------------")
    print("\n启动 FastAPI 服务...")
    print("访问 http://127.0.0.1:8005/docs 可查看自动生成的交互式API文档。")
    
    # 使用 uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8005)