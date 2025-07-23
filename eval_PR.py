import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import ast # 用于安全地解析字符串化的元组
import numpy as np

def plot_pr_curve_from_files(excel_file_path, json_file_path, output_image_path="pr_curve.png"):
    """
    从Excel检索结果和JSON真实标签文件绘制Precision-Recall曲线。

    Args:
        excel_file_path (str): 包含检索结果的Excel/CSV文件路径。
                               预期包含 'pair' (查询和文档的元组字符串) 和 'score' 列。
        json_file_path (str): 包含查询和真实标签的JSON文件路径。
                              预期是 [{adversarial_question: "query", question: "ground_truth_doc"}, ...]
        output_image_path (str): 保存PR曲线图的路径。
    """
    print(f"正在加载 Excel 文件: {excel_file_path}")
    try:
        # 假设Excel文件实际上是CSV格式，由之前的代码生成
        df_results = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"错误：未找到文件 {excel_file_path}。请检查文件路径。")
        return
    except Exception as e:
        print(f"加载 Excel/CSV 文件时发生错误: {e}")
        return

    print(f"正在加载 JSON 文件: {json_file_path}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 构建一个从查询到真实文档的字典，方便查找
        ground_truth_map = {item['adversarial_question']: item['question'] for item in json_data}
    except FileNotFoundError:
        print(f"错误：未找到文件 {json_file_path}。请检查文件路径。")
        return
    except Exception as e:
        print(f"加载 JSON 文件时发生错误: {e}")
        return

    # 解析 'pair' 列，将其从字符串元组转换为实际的元组，并提取 query 和 doc
    # 使用 ast.literal_eval 来安全地解析字符串表示的 Python 字面量（如元组）
    print("正在解析 Excel 文件中的 'pair' 列...")
    # df_results['parsed_pair'] = df_results['pair'].apply(ast.literal_eval)
    # df_results['query'] = df_results['parsed_pair'].apply(lambda x: x[0])
    # df_results['doc'] = df_results['parsed_pair'].apply(lambda x: x[1])

    # 确保 'score' 列是数值类型
    df_results['score'] = pd.to_numeric(df_results['score'], errors='coerce')
    # 移除无法转换为数字的分数行
    df_results.dropna(subset=['score'], inplace=True)
    grouped = df_results.groupby('query')

    # 准备评估所需的数据：所有预测的文档及其相关性标签和分数
    y_true = [] # 真实标签 (0 或 1)
    y_scores = [] # 预测分数
    for query, group in grouped:        
        query = query.strip()
        tmp_df = group.sort_values(by='score', ascending=False).reset_index(drop=True)

        # query = row['query']
        predicted_doc = tmp_df.iloc[0]['doc']
        score = tmp_df.iloc[0]['score']


        # 获取当前查询的真实文档
        ground_truth_doc = ground_truth_map.get(query)

        if ground_truth_doc:
            # 判断预测的文档是否与真实文档相关
            is_relevant = 1 if predicted_doc == ground_truth_doc else 0
            y_true.append(is_relevant)
            y_scores.append(score)
        else:
            # 如果查询在ground truth中不存在，则跳过或按需处理
            print(f"警告: 查询 '{query}' 在JSON真实标签文件中没有找到对应的真实文档。跳过此条。")

    if not y_true or not y_scores:
        print("没有足够的数据来绘制PR曲线。请检查输入文件内容。")
        return
    
    # y_scores, y_true = zip(*sorted(zip(y_scores, y_true), reverse=True))

    # 计算Precision-Recall曲线
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # 对于多条query的情况，需要将所有 (score, is_relevant) 对汇总起来计算PR曲线
    # sklearn.metrics.precision_recall_curve 已经可以处理这种扁平化的列表

    print("正在计算 Precision-Recall 曲线...")
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    print(f"PR AUC: {pr_auc:.4f}")


    # 3. 找到 precision 大于或等于 95% 的所有索引
    # 注意：precision 数组的最后一个元素是 1.0，recall 数组的最后一个元素是 0.0，它们没有对应的阈值。
    # 因此，在匹配阈值时，我们通常会忽略最后一个 precision 值，因为它不对应一个实际的预测行为。
    # precision[:-1] 和 thresholds 是一一对应的。
    target_precision = 0.95
    min_threshold_for_precision = None
    indices = np.where(precision[:-1] >= target_precision)[0]
    if len(indices)>0:
        min_threshold_for_precision = thresholds[indices[0]]

    if min_threshold_for_precision is not None:
        print(f"准确率达到 {target_precision*100:.0f}% 时的阈值: {min_threshold_for_precision:.4f}, recall:{recall[indices[0]]}")
    else:
        print(f"未找到准确率达到 {target_precision*100:.0f}% 的阈值。可能最高准确率未达到此水平。")


    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.grid(True)
    plt.legend(loc="lower left")
    
    # 可选：在图中标记95%准确率的阈值点（如果有找到）
    if min_threshold_for_precision is not None:
        # 找到对应这个阈值的recall和precision
        # 找到第一个分数大于等于阈值的索引
        idx = (thresholds >= min_threshold_for_precision).argmax() 
        # 确保索引有效
        if idx < len(precision):
            plt.plot(recall[idx], precision[idx], 'ro', markersize=8, label=f'Precision={target_precision*100:.0f}% Threshold={min_threshold_for_precision:.4f}')
            plt.vlines(recall[idx], 0, precision[idx], color='red', linestyle='--', lw=1)
            plt.hlines(precision[idx], 0, recall[idx], color='red', linestyle='--', lw=1)
            plt.legend(loc="lower left")


    # 保存图像
    plt.savefig(output_image_path)
    print(f"PR 曲线已保存到: {output_image_path}")
    plt.show() # 显示图表

if __name__ == "__main__":
    # 请根据您实际的文件路径修改这里
    excel_file = './output_search/search_qwen_E06_R06_ck50.xlsx'
    json_file = '/home/liuzihao/intent_detection/panwei_questions.json' # 您的JSON文件路径

    plot_pr_curve_from_files(excel_file, json_file)
