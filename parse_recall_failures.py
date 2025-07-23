#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析log_qwen.txt文件，提取召回失败的记录
"""

import re
import json
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_recall_failures(log_file_path):
    """
    解析日志文件，提取召回失败的记录
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        list: 包含召回失败记录的列表
    """
    failures = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配召回失败的模式
        # 匹配模式：✗ 召回失败 后面跟着查询、期望、实际召回
        pattern = r'✗ 召回失败\s*\n查询: (.+?)\n期望: (.+?)\n实际召回: (.+?)\n'
        
        matches = re.findall(pattern, content, re.DOTALL)
        
        logger.info(f"找到 {len(matches)} 条召回失败记录")
        
        for i, (query, expected, actual) in enumerate(matches, 1):
            failure_record = {
                "id": i,
                "query": query.strip(),
                "expected": expected.strip(),
                "actual_recall": actual.strip()
            }
            failures.append(failure_record)
            
            logger.info(f"解析第 {i} 条失败记录")
            logger.debug(f"查询: {query.strip()[:50]}...")
            
    except FileNotFoundError:
        logger.error(f"文件 {log_file_path} 不存在")
        return []
    except Exception as e:
        logger.error(f"解析文件时出错: {e}")
        return []
    
    return failures

def save_to_json(failures, output_file):
    """
    将召回失败记录保存到JSON文件
    
    Args:
        failures: 召回失败记录列表
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功保存 {len(failures)} 条记录到 {output_file}")
        
    except Exception as e:
        logger.error(f"保存文件时出错: {e}")

def main():
    """
    主函数
    """
    log_file_path = '/home/liuzihao/intent_detection/log.txt'
    output_file = '/home/liuzihao/intent_detection/recall_failures.json'
    
    logger.info("开始解析召回失败记录...")
    
    # 解析召回失败记录
    failures = parse_recall_failures(log_file_path)
    
    if failures:
        # 保存到JSON文件
        save_to_json(failures, output_file)
        
        # 打印统计信息
        logger.info(f"解析完成！共找到 {len(failures)} 条召回失败记录")
        
        # 显示前几条记录作为示例
        if len(failures) > 0:
            logger.info("\n前3条记录示例:")
            for i, failure in enumerate(failures[:3], 1):
                logger.info(f"\n记录 {i}:")
                logger.info(f"查询: {failure['query'][:100]}...")
                logger.info(f"期望: {failure['expected'][:100]}...")
                logger.info(f"实际: {failure['actual_recall'][:100]}...")
    else:
        logger.warning("未找到召回失败记录")

if __name__ == '__main__':
    main()