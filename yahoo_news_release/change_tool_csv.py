import json
import csv

data_file_path = "train.data"
csv_file_path = "test_title_search_in_train_all.csv"

# 读取并解析每一行的JSON数据
with open(data_file_path, 'r') as data_file:
    data = [json.loads(line.strip()) for line in data_file]

# 获取所有可能的字段名
fields = set(field for d in data for field in d.keys())

# 写入CSV文件
with open(csv_file_path, mode='w', newline='',encoding='UTF-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields)

    # 写入表头
    writer.writeheader()

    # 写入数据
    writer.writerows(data)

print(f"CSV file '{csv_file_path}' has been created.")