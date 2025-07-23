import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Any
import pandas as pd

# --- 从 a.py 中复制必要的函数和变量 ---
# `format_instruction` 用于格式化输入文本，使其符合模型期望的格式
def format_instruction(instruction: str, query: str, doc: str) -> str:
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
    return output

# 加载预训练的 tokenizer 和模型
# 注意: 这里的模型是 Qwen/Qwen3-Reranker-0.6B，一个因果语言模型
# 它的输出 logits 是针对词汇表中每个 token 的，我们将利用 "yes" 和 "no" token 的概率来模拟回归输出
tokenizer = AutoTokenizer.from_pretrained("./Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("./Qwen3-Reranker-0.6B") 

# 获取 "no" 和 "yes" token 的 ID，这在 compute_logits 逻辑中至关重要
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

# 定义最大序列长度，与 a.py 保持一致
max_length = 8192

# 定义输入的前缀和后缀 token，用于构建模型输入
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# --- 辅助函数：用于单个样本的 tokenization，不进行填充 ---
def _tokenize_and_add_special_tokens(text: str) -> Dict[str, List[int]]:
    """
    对单个文本进行 tokenization 并添加前缀/后缀特殊 token，但不进行填充。
    返回 token ID 列表和注意力掩码列表。
    """
    inputs = tokenizer(
        text,
        padding=False,  # 在这里不进行填充
        truncation='longest_first',
        return_attention_mask=True,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens) # 确保预留空间
    )
    
    # 添加前缀和后缀 token
    input_ids = prefix_tokens + inputs['input_ids'] + suffix_tokens
    attention_mask = [1] * len(prefix_tokens) + inputs['attention_mask'] + [1] * len(suffix_tokens)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

# --- PyTorch Dataset 定义 ---
class RegressionDataset(torch.utils.data.Dataset):
    """
    一个自定义的 PyTorch Dataset 类，用于处理回归任务的数据。
    它将查询、文档和真实分数打包成模型训练所需的格式。
    """
    def __init__(self, queries: List[str], documents: List[str], true_scores: np.ndarray, task_instruction: str):
        # 格式化查询-文档对
        self.pairs = [format_instruction(task_instruction, query, doc) for query, doc in zip(queries, documents)]
        # 存储真实的回归标签（分数）
        self.true_scores = true_scores

    def __len__(self) -> int:
        # 返回数据集中样本的数量
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        根据索引获取单个样本。
        返回一个字典，包含原始 token ID 列表、注意力掩码列表和对应的真实回归标签。
        注意：这里返回的是列表，而不是已经填充的张量。填充将在 Data Collator 中完成。
        """
        pair = self.pairs[idx]
        # 调用辅助函数获取 token ID 列表和注意力掩码列表
        encoded_input = _tokenize_and_add_special_tokens(pair) 
        
        return {
            "input_ids": encoded_input["input_ids"],
            "attention_mask": encoded_input["attention_mask"],
            "labels": self.true_scores[idx]
        }

# --- 自定义 Data Collator ---
class CustomDataCollator:
    """
    一个自定义的 Data Collator。
    它接收来自 RegressionDataset.__getitem__ 的原始 token ID 列表和注意力掩码列表，
    并负责对整个批次进行填充，确保所有序列长度一致。
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 提取批次中所有样本的 input_ids 和 attention_mask 列表
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float32)

        # 使用 tokenizer 对整个批次进行填充
        # padding=True 会填充到批次中最长的序列长度
        # return_tensors="pt" 会返回 PyTorch 张量
        padded_batch = tokenizer.pad(
            {'input_ids': input_ids_list, 'attention_mask': attention_mask_list},
            padding=True,
            return_tensors="pt",
            max_length=max_length # 确保填充不超过全局最大长度
        )
        
        batch = {
            "input_ids": padded_batch["input_ids"],
            "attention_mask": padded_batch["attention_mask"],
            "labels": labels
        }
        return batch

# --- `process_inputs` 函数 (仅用于推理时的批处理，不被 Dataset 调用) ---
# 这个函数保持原样，用于在训练或推理时处理一批原始字符串，并进行完整的 tokenization 和填充。
def process_inputs(pairs: List[str]) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        pairs, 
        padding=True, # 这里直接进行填充到批次最长或 max_length
        truncation='longest_first', 
        return_attention_mask=True, 
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens) # 预留空间
    )
    # 添加前缀和后缀 token
    # 注意：这里需要确保 inputs['input_ids'] 和 inputs['attention_mask'] 都是列表才能进行拼接
    # 如果 tokenizer 返回的是 Tensor，需要先转换为列表再拼接
    processed_input_ids = []
    processed_attention_mask = []
    for i in range(len(inputs['input_ids'])):
        current_input_ids = inputs['input_ids'][i]
        current_attention_mask = inputs['attention_mask'][i]

        processed_input_ids.append(prefix_tokens + current_input_ids + suffix_tokens)
        processed_attention_mask.append([1] * len(prefix_tokens) + current_attention_mask + [1] * len(suffix_tokens))

    # 再次填充，因为添加了前缀和后缀后长度可能变化，且需要统一为 Tensor
    final_inputs = tokenizer.pad(
        {'input_ids': processed_input_ids, 'attention_mask': processed_attention_mask},
        padding=True,
        return_tensors="pt",
        max_length=max_length # 确保最终填充到全局最大长度
    )
    return final_inputs


# --- 自定义 Trainer 类 ---
class CustomRegressionTrainer(Trainer):
    """
    一个自定义的 Trainer 类，重写了 `compute_loss` 方法以实现特定的回归损失计算逻辑。
    该逻辑与 a.py 中的 `compute_logits` 函数保持一致。
    """
    # 更改了 compute_loss 的签名以接受 **kwargs
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, **kwargs) -> torch.Tensor:
        """
        计算模型的回归损失。
        Args:
            model: 要训练的模型。
            inputs: 包含输入 token ID、注意力掩码和真实标签的字典。
            return_outputs: 如果为 True，则返回损失和模型输出。
            **kwargs: 接受 Trainer 传递的任何额外关键字参数（例如 'num_items_in_batch'）。
        Returns:
            损失值。
        """
        # 从输入中弹出真实标签
        labels = inputs.pop("labels")
        
        # 模型前向传播，获取 logits
        outputs = model(**inputs)
        logits = outputs.logits

        # --- 还原 a.py 中 compute_logits 的逻辑来获取回归预测值 ---
        # 获取序列中最后一个 token 的 logits
        # 假设最后一个 token 的 logits 包含了对整个序列的最终决策信息
        batch_scores = logits[:, -1, :] 
        
        # 检查 token_true_id 和 token_false_id 是否在词汇表范围内
        # 这是为了防止索引越界错误
        if token_true_id >= batch_scores.shape[1] or token_false_id >= batch_scores.shape[1]:
            raise ValueError(f"Error: token_true_id ({token_true_id}) or token_false_id ({token_false_id}) "
                             f"is out of vocabulary range ({batch_scores.shape[1]}). "
                             "This might indicate a mismatch between tokenizer and model, or an invalid token ID.")

        # 提取 "yes" 和 "no" token 对应的分数
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        
        # 将这两个分数堆叠起来，并应用 log_softmax 转换为概率对数
        stacked_scores = torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = F.log_softmax(stacked_scores, dim=1)
        
        # 获取 "yes" token 对应的概率 (exp(log_softmax_score_for_yes))
        # 这就是我们将其视为回归预测的连续值，范围在 0 到 1 之间
        predicted_scores = log_softmax_scores[:, 1].exp()

        # 计算回归损失 (均方误差 MSE)
        # 确保预测值和标签的维度匹配
        loss = F.mse_loss(predicted_scores.squeeze(), labels.squeeze())

        return (loss, outputs) if return_outputs else loss

# --- 模拟数据 (替换为你的实际回归数据) ---
df = pd.read_excel('./reranker_output_multi_process.xlsx')
print(f"{df.shape}, {df.columns}")
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df.dropna(subset=['score'], inplace=True)
print(f"{df.shape}, {df.columns}")

# 假设您的回归任务是根据查询和文档来预测一个相关性分数，分数在 0 到 1 之间。
task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
queries_data = df['query']
documents_data = df['doc']
# 模拟对应的真实回归标签 (例如，相关性分数)
# 这些是您模型应该学习预测的目标连续值
true_scores_data = np.array(df['score'], dtype=np.float32)

# 创建训练数据集
train_dataset = RegressionDataset(queries_data, documents_data, true_scores_data, task_instruction)

# --- 训练参数配置 ---
training_args = TrainingArguments(
    output_dir="./regression_model_output", # 模型输出和检查点保存目录
    num_train_epochs=1, # 训练的轮数
    per_device_train_batch_size=4, # 每个设备（GPU）的训练批次大小
    warmup_steps=10, # 预热步数，用于学习率调度
    weight_decay=0.01, # 权重衰减（L2 正则化）
    logging_dir="./logs", # 日志目录
    logging_steps=10, # 每隔多少步记录一次日志
    gradient_accumulation_steps=1, # 梯度累积步数，用于模拟更大批次
    # eval_strategy="epoch", # 如果有验证集，可以开启每个 epoch 评估
    # save_strategy="epoch", # 如果有验证集，可以开启每个 epoch 保存模型
    load_best_model_at_end=False, # 如果有验证集，可以开启在训练结束时加载最佳模型
    report_to="none", # 不报告到任何外部集成 (例如 WandB)
)

# --- 实例化自定义 Trainer ---
trainer = CustomRegressionTrainer(
    model=model, # 传入预训练模型
    args=training_args, # 传入训练参数
    train_dataset=train_dataset, # 传入训练数据集
    tokenizer=tokenizer, # 传入 tokenizer，Trainer 可能需要它进行内部处理
    data_collator=CustomDataCollator(), # 传入自定义的 data collator
)

# --- 开始训练 ---
print("开始大语言模型回归再训练...")
try:
    trainer.train()
    print("训练完成！模型已保存到指定目录。")
    # --- 添加保存模型的代码 ---
    # 保存最终训练好的模型到 output_dir 指定的路径
    trainer.save_model(training_args.output_dir)
    print(f"模型权重已保存到: {training_args.output_dir}")
except Exception as e:
    print(f"训练过程中发生错误: {e}")

# --- 训练后模型推理示例 ---
print("\n--- 训练后模型推理示例 ---")
# 假设你有一个新的查询和文档对，想预测其回归分数
new_query_for_inference = "What is the capital of France?"
new_doc_for_inference = "Paris is the capital and most populous city of France."
new_pair_for_inference = format_instruction(task_instruction, new_query_for_inference, new_doc_for_inference)

# 准备输入，使用原始文本列表
test_inputs_raw = [new_pair_for_inference]
# 使用 `process_inputs` 将原始文本转换为模型输入格式
test_inputs = process_inputs(test_inputs_raw)

# 将输入数据移动到模型所在的设备 (CPU 或 GPU)
for key in test_inputs:
    if isinstance(test_inputs[key], torch.Tensor):
        test_inputs[key] = test_inputs[key].to(model.device)

# 进行预测 (在评估模式下，并且不计算梯度)
with torch.no_grad():
    model.eval() # 切换到评估模式
    outputs = model(**test_inputs)
    logits = outputs.logits

    # 还原 a.py 中 compute_logits 的逻辑来获取预测值
    batch_scores = logits[:, -1, :] # 获取最后一个 token 的 logits

    # 再次检查 token ID 范围
    if token_true_id >= batch_scores.shape[1] or token_false_id >= batch_scores.shape[1]:
        print(f"警告: token_true_id ({token_true_id}) 或 token_false_id ({token_false_id}) "
              f"超出词汇表范围 ({batch_scores.shape[1]})。可能无法进行有效预测。")
        predicted_score = float('nan') # 设置为 NaN 或其他错误值
    else:
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        stacked_scores = torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = F.log_softmax(stacked_scores, dim=1)
        predicted_score = log_softmax_scores[:, 1].exp().item() # 获取单个预测值

print(f"\n对于新输入 (查询: '{new_query_for_inference}', 文档: '{new_doc_for_inference}')，")
print(f"模型预测的回归分数是: {predicted_score:.4f}")
