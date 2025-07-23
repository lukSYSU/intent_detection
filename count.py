from pymilvus import Collection

from pymilvus import connections, Collection
import logging
import sys

# 配置日志记录
def setup_logging():
    """设置日志记录，同时输出到控制台和文件"""
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件handler
    file_handler = logging.FileHandler('count_log.txt', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

try:
    # 首先建立与Milvus的连接
    logger.info("正在连接到Milvus服务器...")
    connections.connect("default", host="localhost", port="19530")
    logger.info("成功连接到Milvus服务器")
    
    # 加载你的 Collection
    logger.info("正在加载集合...")
    collection = Collection("qwen3_embedding")
    logger.info(f"成功加载集合: {collection.name}")
    
    # 在 flush 之后获取实体数量
    logger.info("正在刷新集合数据...")
    collection.flush() # 再次确保数据已落盘
    logger.info(f"集合中的实体总数: {collection.num_entities}")
    
    # 关闭连接
    connections.disconnect("default")
    logger.info("已关闭与Milvus的连接")
    
except Exception as e:
    logger.error(f"发生错误: {e}")
    # 确保关闭连接
    try:
        connections.disconnect("default")
        logger.info("已关闭与Milvus的连接")
    except:
        pass