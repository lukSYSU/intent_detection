from elasticsearch import Elasticsearch
import json

def show_table_record_num(es_connection_args):
    # 连接到您的ES实例
    try:
        es_client = Elasticsearch(**es_connection_args)

        # 验证连接
        if not es_client.ping():
            raise ConnectionError("无法连接到Elasticsearch服务。")

        print("成功连接到Elasticsearch！\n")

        # 使用 Cat API 获取索引信息
        # format="json" 会返回结构化的数据，比纯文本更容易处理
        indices_info = es_client.cat.indices(format="json", v=True)

        if not indices_info:
            print("您的Elasticsearch中还没有任何索引。")
        else:
            print("索引名称           | 文档数量")
            print("--------------------|-----------")
            for index in indices_info:
                index_name = index['index']
                doc_count = index['docs.count']
                print(f"{index_name:<20}| {doc_count}")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'es_client' in locals() and es_client:
            es_client.close()


def show_table_fields(es_conn_args, index_name):
    es_client = None  

    try:
        es_client = Elasticsearch(**es_conn_args)
        
        # 验证连接
        if not es_client.ping():
            raise ConnectionError("无法连接到Elasticsearch服务。")

        print("成功连接到Elasticsearch！\n")
        
        # 检查索引是否存在
        if not es_client.indices.exists(index=index_name):
            print(f"错误：索引 '{index_name}' 不存在。")
        else:
            print(f"正在获取索引 '{index_name}' 的字段结构 (Mapping)...")
            
            # 获取映射(Mapping)
            mapping_response = es_client.indices.get_mapping(index=index_name)
            
            # mapping_response 的结构是 { 'index_name': { 'mappings': { ... } } }
            properties = mapping_response[index_name]['mappings']['properties']
            
            print("\n" + "="*50)
            print(f"索引 '{index_name}' 的字段列表:")
            print("="*50)
            
            # 遍历并以清晰的格式打印每个字段及其类型
            for field, details in properties.items():
                field_type = details.get('type', 'object') # 获取字段类型
                
                if field_type == 'dense_vector':
                    dims = details.get('dims', '未知')
                    print(f"- 字段名: {field:<20} 类型: {field_type} (维度: {dims})")
                elif field_type == 'object' and 'properties' in details:
                    # 处理像 metadata 这样的嵌套字段
                    print(f"- 字段名: {field:<20} 类型: {field_type}")
                    for sub_field, sub_details in details['properties'].items():
                        print(f"    - 子字段: {sub_field:<16} 类型: {sub_details.get('type', '未知')}")
                else:
                    print(f"- 字段名: {field:<20} 类型: {field_type}")
            
            print("="*50)

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保客户端连接被关闭
        if es_client:
            es_client.close()


if __name__ == '__main__':
    # Elasticsearch 的连接参数
    es_conn_args = {
        "hosts": "http://localhost:9200"
        # 如果ES需要认证，可以添加 es_user 和 es_password
        # "es_user": "elastic",
        # "es_password": "your_password"
    }

    show_table_record_num(es_conn_args)

    index_name = "qwen3_panwei_index"
    show_table_fields(es_conn_args, index_name)