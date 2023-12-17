# 计算隐写文本中到底包含了多少秘密信息
import time
import torch
import json
import os
import numpy as np
import scipy.io as sio
import argparse
import csv
import codecs
import jsonlines
import gensim.downloader as api
import pickle

# 定义了不同词嵌入名称和对应模型文件
word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
}

# 该函数用于获取包含秘密信息的隐写文本，它从指定的CSV文件中读取每一行，提取关键词，并将关键词映射为预先编码的标识符enc_dict
def get_keywordsets_bitstream_jsonl_wo_unk_v2(file_name="/data2/yahoo_news_release/test_title_search_in_dev_all.csv", enc_dict={}):
    # 这里秘密信息的最开始5bit表示秘密信息index的数量 方便后续解码
    keyword_sets = []
    all_stext = []
    with open(file_name, "r", encoding="utf-8") as f:
        for row in jsonlines.Reader(f):
            keywords = row['keywords']
            if keywords != '':
                keywords = list(keywords.split())
            else:
                keywords = []
            flag = 1
            for keyword in keywords:
                if keyword not in enc_dict.keys():
                    flag = 0
                    print(keyword)
            if flag:
                stext = row['stext']
                all_stext.append(stext)

    return all_stext


embedding = "glove"  # 指定要使用的嵌入模型
file_name = "/data2/yahoo_news_release/test_title_search_in_train_all.jsonl"  # 指定包含隐写文本信息的文件路径
save_path_dict = "/home/blockchain/wanghl/train_lm_with_keywords/data/dict_wo_unktest_title_search_in_train_allglove.pkl"  # 指定预先编码的字典的文件路径
with open(save_path_dict, 'rb') as file:  # 加载预先编码的字典
    enc_dict = pickle.load(file)
all_bit_nums = 0
all_text = get_keywordsets_bitstream_jsonl_wo_unk_v2(file_name=file_name, enc_dict=enc_dict)  # 获取包含预先编码关键词的隐写文本
bit_num = 0
all_letter_num = 0
for tmp in all_text:
    words = tmp.split(" ")
    for word in words:
        all_letter_num += len(word)
print(all_letter_num)  # 统计并打印文本隐写中所有字母的数量
all_bit_nums = all_letter_num*8
print(all_bit_nums)  # 计算并打印所有字母对应的比特数
all_stego_length = 0  # 统计指定文件中所有隐写文本的长度
stego_file = '/home/blockchain/wanghl/train_lm_with_keywords/result_randomseed/generate_keyword/max_ac_gpt2_train_all_2022.11.22-11:03:49_5.0.jsonl'
with open(stego_file, "r", encoding="utf-8") as sf:
    for row in jsonlines.Reader(sf):
        if 'stego' in row.keys():
            stgeo = row['stego']
            all_stego_length += len(stgeo.split(" "))
print(all_stego_length)

print(all_bit_nums/all_stego_length)  # 打印每个字母对应的平均比特数
