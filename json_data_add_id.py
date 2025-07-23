import json

def add_ids_to_json(input_file_path, output_file_path):
    """
    读取一个JSON文件，为列表中的每个对象添加一个唯一的ID，并写入新文件。

    :param input_file_path: str, 输入的JSON文件路径。
    :param output_file_path: str, 输出的JSON文件路径。
    """
    try:
        # 使用'utf-8'编码读取原始JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保数据是一个列表
        if not isinstance(data, list):
            print("错误：JSON文件内容不是一个列表。")
            return

        # 遍历列表中的每一个对象（字典），并添加一个ID
        # enumerate(data, 1) 会生成 (1, item1), (2, item2), ...
        for item_id, item in enumerate(data, 1):
            # 为了保持原有的顺序，我们还是直接在原字典上操作
            item.update({'id': item_id}) # 虽然id会加在最后
            
        # 让id在前面
        data_with_ids = [{'id': i + 1, **item} for i, item in enumerate(data)]


        # 使用'utf-8'编码写入新的JSON文件
        # ensure_ascii=False 确保中文字符能正常显示
        # indent=2 使输出的JSON文件格式化，易于阅读
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_with_ids, f, ensure_ascii=False, indent=2)

        print(f"处理成功！已为数据添加ID，并保存到文件: {output_file_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file_path} 的JSON格式不正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == '__main__':
    # 定义输入和输出文件路径
    input_file = "./questions_adversarial_panwei.json"
    output_file = "questions_adversarial_withid_panwei.json"

    # 调用函数
    add_ids_to_json(input_file, output_file)

