# -*- coding: utf-8 -*-
from datasets import load_dataset
import os

def download_qqp_with_hf_datasets():
    """
    使用 Hugging Face datasets 库下载 QQP 数据集。
    这是最简单、最标准的方式。
    """
    print("--- 开始使用 Hugging Face `datasets` 下载 QQP ---")
    
    # 加载 GLUE 基准中的 QQP 任务
    # 库会自动处理下载和缓存
    print("正在加载 'glue', 'qqp' 数据集...")
    custom_cache_dir = "./my_huggingface_cache"
    print(f"\n*注意*: 数据集将被缓存到指定目录: {os.path.abspath(custom_cache_dir)}")

    qqp_dataset = load_dataset('glue', 'qqp', cache_dir=custom_cache_dir)
    print("加载完成！")

    # `load_dataset` 返回一个 DatasetDict 对象，通常包含'train', 'validation', 'test'
    print("\n数据集结构:")
    print(qqp_dataset)
    # 输出示例:
    # DatasetDict({
    #     train: Dataset({
    #         features: ['question1', 'question2', 'label', 'idx'],
    #         num_rows: 363846
    #     })
    #     validation: Dataset({
    #         features: ['question1', 'question2', 'label', 'idx'],
    #         num_rows: 40430
    #     })
    #     test: Dataset({
    #         features: ['question1', 'question2', 'label', 'idx'],
    #         num_rows: 390965
    #     })
    # })
    
    print(f"------------------------------------------")
    # 查看一条训练数据
    print("\n查看一条训练数据示例:")
    example = qqp_dataset['train'][0]
    print(example)
    # 输出示例:
    # {
    #  'question1': 'How is the life of a math student? Could you describe your own experiences?',
    #  'question2': 'Which level of math gets you the most girls? And how much time does it take to get that level?',
    #  'label': 0,  # 0代表不重复, 1代表重复
    #  'idx': 0
    # }
    
    print("-" * 50)



if __name__ == '__main__':
    download_qqp_with_hf_datasets()
    


