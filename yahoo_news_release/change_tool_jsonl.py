import json

data_file_path = "train.data"
jsonl_file_path = "test_title_search_in_train_all.jsonl"

# 读取并逐行写入 JSONL 文件
with open(data_file_path, 'r', encoding='utf-8') as data_file, \
        open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
    for line in data_file:
        # 解析 JSON 数据
        json_data = json.loads(line.strip())

        # 将 JSON 数据写入 JSONL 文件
        jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

print(f"JSONL file '{jsonl_file_path}' has been created.")