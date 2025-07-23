import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置
EXCEL_PATH = 'questions.xlsx'
RESULT_EXCEL_PATH = 'questions_adversarial.xlsx'
API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
MODEL = 'deepseek-v3'

# 读取API KEY
with open('apikeys-ali.txt', 'r', encoding='utf-8') as f:
    api_keys = [line.strip() for line in f if line.strip()]

if not api_keys:
    raise ValueError("apikeys-ali.txt 没有有效API KEY")

api_key_lock = threading.Lock()
api_key_iter = iter(api_keys * 10000)  # 无限轮询

def get_api_key():
    with api_key_lock:
        return next(api_key_iter)

def build_adversarial_prompt(question):
    return (
        # "请以语义保持但表达复杂化、增加歧义和知识稀疏的方式，改写下列问题，使其对自动理解和信息抽取模型更具挑战性。"
        # "不要简单同义改写，要在结构、指代、修饰语等层面进行变化，但不可更改核心语义。直接输出改写问题即可，不要添加任何额外的修饰。原始问题："
        "请将下列表述规范、偏向业务术语的投诉问题，改写为普通用户从自身角度、用自然口语表达提出的问题。要求："
        "- 保持核心语义不变，不要遗漏或增添事实,尤其是不要遗漏关键参数。"
        "- 用第一人称或用户视角，体现用户的真实困扰或需求。"
        "- 不要简单替换词语，要整体调整为用户常用的提问方式。"
        "- 只输出改写后的用户口吻问题。不要添加'问题改写版本:'这种无关的话"
        "- 输出不需要用引号“”或""包裹"
        "原始问题："
        f"\n\"{question}\"\n"
    )

def query_deepseek(prompt, api_key, max_tokens=256, temperature=0.8, retry=3):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": MODEL,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    }
    for attempt in range(retry):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=60)
            if resp.ok:
                # 返回结构根据API规范，如果output/text找不到请调整为output/choices/0/message/content
                try:
                    return resp.json()["output"]["choices"][0]["message"]["content"]
                except Exception:
                    return resp.json()["output"]["text"]  # 或根据实际返回调整
            else:
                print(f"[{api_key[:8]}...]接口异常：{resp.text}")
                if resp.status_code in (429, 401):
                    api_key = get_api_key()
                time.sleep(2)
        except Exception as e:
            print(f"请求失败: {e}")
            time.sleep(2)
    return None


def process_row(idx, question):
    api_key = get_api_key()
    prompt = build_adversarial_prompt(question)
    adv_question = query_deepseek(prompt, api_key)
    return idx, adv_question

# 读取问题
df = pd.read_excel(EXCEL_PATH)
if "question" not in df.columns:
    raise ValueError("输入表格需要有question列")

results = [None] * len(df)

with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
    futures = [executor.submit(process_row, idx, q) for idx, q in enumerate(df["question"])]
    for future in as_completed(futures):
        idx, adv = future.result()
        results[idx] = adv
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"进度: {idx + 1}/{len(df)}")

df["adversarial_question"] = results
df.to_excel(RESULT_EXCEL_PATH, index=False)
print("全部完成，已保存至", RESULT_EXCEL_PATH)
