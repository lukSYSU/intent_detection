#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Excel文件中提取"用户问题"和"adversarial_question"两列，
将"用户问题"重命名为"question"并保存为JSON文件
"""

import pandas as pd
import json
import os

def extract_excel_to_json(excel_file_path, output_json_path):
    """
    从Excel文件中提取指定列并保存为JSON
    
    Args:
        excel_file_path (str): Excel文件路径
        output_json_path (str): 输出JSON文件路径
    """
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        
        # 显示列名以便调试
        print(f"Excel文件中的列名: {list(df.columns)}")
        
        # 检查所需列是否存在
        required_columns = ['用户问题', 'adversarial_question']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"错误: 缺少以下列: {missing_columns}")
            return False
        
        # 提取所需列
        extracted_data = df[required_columns].copy()
        
        # 重命名"用户问题"为"question"
        extracted_data = extracted_data.rename(columns={'用户问题': 'question'})
        
        # 删除包含空值的行
        extracted_data = extracted_data.dropna()
        
        # 转换为字典列表
        data_list = extracted_data.to_dict('records')
        
        # 保存为JSON文件
        print(f"正在保存到JSON文件: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"成功提取 {len(data_list)} 条记录")
        print(f"数据已保存到: {output_json_path}")
        
        # 显示前几条记录作为示例
        if data_list:
            print("\n前3条记录示例:")
            for i, record in enumerate(data_list[:3]):
                print(f"记录 {i+1}: {record}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {excel_file_path}")
        return False
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

def main():
    # 文件路径
    excel_file = "/home/liuzihao/intent_detection/panwei_5000_questions_adversarial.xlsx"
    json_file = "/home/liuzihao/intent_detection/extracted_questions.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(excel_file):
        print(f"错误: Excel文件不存在: {excel_file}")
        return
    
    # 执行提取
    success = extract_excel_to_json(excel_file, json_file)
    
    if success:
        print("\n提取完成!")
    else:
        print("\n提取失败!")

if __name__ == "__main__":
    main()