import json
import torch
import transformers
from pymilvus import MilvusClient, DataType
import config as cfgMgmt

# --- 初始化 Milvus Client 和 Embedding 模型 ---

print(f"尝试连接到 Milvus 客户端: {cfgMgmt.MILVUS_URI}...")
milvus_client = MilvusClient(uri=cfgMgmt.MILVUS_URI, **cfgMgmt.MILVUS_CONN_ARGS)
print("Milvus 客户端连接成功。")

print(f"正在加载 Embedding 模型 ({cfgMgmt.EMBEDDING_MODEL_PATH})...")
# create_embedding_model 返回的是 LangChain 的 HuggingFaceEmbeddings 对象
# 为了用 PyMilvus，我们需要直接使用 HuggingFace pipeline 或 AutoModel
from transformers import AutoTokenizer, AutoModel
embedding_tokenizer = AutoTokenizer.from_pretrained(cfgMgmt.EMBEDDING_MODEL_PATH)
embedding_model_py = AutoModel.from_pretrained(cfgMgmt.EMBEDDING_MODEL_PATH).to(cfgMgmt.DEVICE).eval()
print("Embedding 模型加载成功。")



def get_text_embedding(text: str) -> list:
    """
    使用加载的 Qwen3-Embedding 模型为文本生成嵌入向量。
    """
    inputs = embedding_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(cfgMgmt.DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = embedding_model_py(**inputs).last_hidden_state[:, 0].squeeze().cpu().numpy().tolist()
    return embeddings

def find_ids_by_text_content(text_to_find: str, collection_name: str, top_k: int = 5) -> list:
    """
    根据文本内容，通过向量搜索在 Milvus 中查找并返回匹配向量的 ID 列表。
    """
    if not milvus_client:
        print("Milvus 客户端未初始化。")
        return []

    print(f"\n正在查找文本: '{text_to_find}' 对应的向量 ID...")

    try:
        # 将目标文本转换为向量
        query_vector = get_text_embedding(text_to_find)

        # 加载 Collection 以便搜索 (如果 Collection 尚未加载)
        # 确保 Collection 存在并已加载
        if not milvus_client.has_collection(collection_name=collection_name):
            print(f"Collection '{collection_name}' 不存在。")
            return []
        
        # 搜索，并指定 output_fields 以获取文本内容和主键 ID
        # 默认情况下，LangChain Milvus 集成会创建 'text' 字段存储内容，
        # 和 'pk' 或 'id' 字段作为主键。
        # 我们假设你的主键字段名为 'pk'，文本内容字段名为 'text'。
        # 你可能需要根据实际Collection的Schema调整这里。
        
        # 确认 Collection 已加载
        status = milvus_client.get_load_state(collection_name=collection_name)
        if status.state != 'Loaded':
            print(f"Collection '{collection_name}' 未加载，正在加载...")
            milvus_client.load_collection(collection_name=collection_name)
            # 等待加载完成，实际应用中可能需要更健壮的等待机制
            import time
            time.sleep(2) # 简单等待2秒，确保加载完成

        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector], # 传入查询向量
            limit=top_k,         # 返回最相似的 top_k 个结果
            output_fields=["text", "pk"], # 确保返回文本内容和主键ID
            # 可以在这里添加 filter，例如 "text like '%partial_text%'"
            # 但精确匹配通常需要从返回结果中过滤
        )

        found_ids = []
        print(f"相似度搜索结果 (Top {top_k}):")
        
        # search_res 是一个列表的列表，外层列表对应查询数量 (这里是1个查询)
        for hits in search_res:
            for hit in hits:
                # hit.entity 是一个字典，包含 output_fields 中指定的数据
                doc_content = hit.entity.get("text")
                doc_id = hit.entity.get("pk") # 或者你的主键字段名，例如 'id'

                print(f"  - ID: {doc_id}, 相似度: {hit.score:.4f}, 内容: '{doc_content[:100]}...'")

                # 精确匹配文本内容，确保删除的是你想要的
                if doc_content and doc_id is not None and doc_content.strip() == text_to_find.strip():
                    found_ids.append(doc_id)
                    print(f"    -> 找到精确匹配，ID为: {doc_id}")
        
        if not found_ids:
            print(f"未找到与文本 '{text_to_find}' 精确匹配的向量ID。")
        
        return found_ids

    except Exception as e:
        print(f"查找向量 ID 失败: {e}")
        return []

def delete_vectors_by_ids_pymilvus(ids_to_delete: list, collection_name: str):
    """
    使用 PyMilvus 客户端根据 ID 列表删除 Milvus 中的向量。
    """
    if not milvus_client:
        print("Milvus 客户端未初始化。无法执行删除。")
        return

    if not ids_to_delete:
        print("没有要删除的 ID。")
        return

    print(f"\n正在删除 ID 为 {ids_to_delete} 的向量...")
    try:
        # 使用 pymilvus.MilvusClient.delete 方法
        res = milvus_client.delete(
            collection_name=collection_name,
            pks=ids_to_delete # 传入包含要删除 ID 的列表
        )
        print(f"成功删除了 ID 为 {ids_to_delete} 的向量。")
        print(f"删除结果: {res}")
    except Exception as e:
        print(f"删除向量失败: {e}")


if __name__ == "__main__":

    text_to_delete = "苹果公司（Apple Inc.）是美国一家高科技公司。由史蒂夫·乔布斯、斯蒂夫·沃兹尼亚克和罗纳德·韦恩等人于1976年4月1日创立。"

    # 1. 查找文本对应的 ID
    # 这里 top_k 可以根据你对匹配的预期数量进行调整
    found_ids = find_ids_by_text_content(text_to_delete, cfgMgmt.COLLECTION_NAME, top_k=5)

    # 2. 如果找到了 ID，则执行删除
    if found_ids:
        delete_vectors_by_ids_pymilvus(found_ids, cfgMgmt.COLLECTION_NAME)
    else:
        print(f"未找到要删除的文本 '{text_to_delete}' 对应的向量。")

    # 验证是否删除成功（可选）：尝试再次查找已删除的文本
    print("\n尝试再次查找已删除的文本，验证是否仍在：")
    remaining_ids = find_ids_by_text_content(text_to_delete, cfgMgmt.COLLECTION_NAME, top_k=1)
    if not remaining_ids:
        print(f"文本 '{text_to_delete}' 已成功从 Milvus 中删除。")
    else:
        print(f"警告: 文本 '{text_to_delete}' 仍存在，ID: {remaining_ids}")