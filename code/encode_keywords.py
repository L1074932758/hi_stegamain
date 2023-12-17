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
import argparse


word_embedding = {
    'glove': "glove-wiki-gigaword-300",
    'word2vec': "word2vec-google-news-300"
}

# 从包含关键词JSON Lines格式的文件中创建嵌入字典
# 加载预训练的词嵌入模型，将关键词映射到对应的嵌入向量，生成的词典以Pickle形式保存
def create_enc_dict_jsonl(file_name, embedding):
    print("create_enc_dict......")
    embedding_file = word_embedding[embedding]
    print('file_name: ', file_name)
    print('word_embedding: ', embedding)

    # Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    # encoder = api.load(embedding_file)
    encoder = api.load("glove-wiki-gigaword-300")
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for row in jsonlines.Reader(f):
            keywords = row['keywords']
            if keywords != '':
                keywords = list(keywords.split())
            else:
                keywords = []

            if len(keywords) != 0:
                for word in keywords:
                    if word in encoder.index_to_key:
                        print(word)
                        glove_dict[word] = encoder[word]
                    else:
                        glove_dict[word] = encoder['unk']
                        print(word)
        print('{} keyword embeddings done'.format(embedding))

    save_path_dict = os.path.join(
        "data", 'dict_' + file_name.split("/")[-1].split(".")[-2] + embedding + '.pkl')
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f, pickle.HIGHEST_PROTOCOL)

# 与上述处理类似，但是在处理未知词（unk)时不会添加默认的未知词向量
def create_enc_dict_jsonl_wo_unk(file_name, embedding):
    print("create_enc_dict......")
    embedding_file = word_embedding[embedding]
    print('file_name: ', file_name)
    print('word_embedding: ', embedding)

    # Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    #encoder = api.load(embedding_file)
    encoder = api.load("glove-wiki-gigaword-300")
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for row in jsonlines.Reader(f):
            keywords = row['keywords']
            if keywords != '':
                keywords = list(keywords.split())
            else:
                keywords = []

            if len(keywords) != 0:
                for word in keywords:
                    if word in encoder.index_to_key:
                        glove_dict[word] = encoder[word]
                    else:
                        #glove_dict[word] = encoder['unk']
                        print(word)
        print('{} keyword embeddings done'.format(embedding))

    save_path_dict = os.path.join(
        "data", 'dict_wo_unk' + file_name.split("/")[-1].split(".")[-2] + embedding + '.pkl')
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f, pickle.HIGHEST_PROTOCOL)

# 从CSV格式的文件中创建嵌入字典，同上操作保存生成的字典
def create_enc_dict(file_name, embedding):
    print("create_enc_dict......")
    embedding_file = word_embedding[embedding]
    print('file_name: ', file_name)
    print('word_embedding: ', embedding)

    # Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    #encoder = api.load(embedding_file)
    encoder = api.load("glove-wiki-gigaword-300")
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}

    csv.field_size_limit(500 * 1024 * 1024)
    with codecs.open(file_name, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            keywords = row['keywords']
            if keywords != '':
                keywords = list(keywords.split())
            else:
                keywords = []
                print(keywords)
            if len(keywords) != 0:
                for word in keywords:
                    if word in encoder.index_to_key:
                        print(word)
                        glove_dict[word] = encoder[word]
                    else:
                        glove_dict[word] = encoder['unk']
        print('{} keyword embeddings done'.format(embedding))

    save_path_dict = os.path.join(
        "data", 'dict_' + file_name.split("/")[-1].split(".")[-2] + embedding + '.pkl')
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f, pickle.HIGHEST_PROTOCOL)

# 根据给定的任务类型，从不同的文件格式创建嵌入字典。
# 根据任务类型选择相应的文件读取方式，并将关键词映射到对应的嵌入向量，最后保存生成的字典
def create_enc_dict_ori(file_name, embedding, task):
    print("create_enc_dict......")

    embedding_file = word_embedding[embedding]
    if task == 'key2article':
        folder_name = file_name
    else:
        folder_name = os.path.dirname(file_name)

    print('file_name: ', file_name)
    print('folder_name: ', folder_name)
    print('word_embedding: ', embedding)

    # Load word embedding data
    print('{} word embeddings loading...'.format(embedding))
    encoder = api.load(embedding_file)
    print('{} word embeddings loaded'.format(embedding))
    glove_dict = {}
    if task == 'commentgen':
        csv.field_size_limit(500 * 1024 * 1024)
        with codecs.open(file_name, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                keywords = row['keywords']
                if keywords != '':
                    keywords = list(keywords.split())
                else:
                    keywords = []
                print(keywords)
                if len(keywords) != 0:
                    for word in keywords:
                        print(word)
                        glove_dict[word] = encoder[word]
        print('{} keyword embeddings done'.format(embedding))
    elif not task == 'key2article':
        file1 = open(file_name, "r+")
        lines = file1.readlines()

        i = 0
        for line in lines:
            keywords = list(line.strip().split(", "))
            print(keywords)
            for word in keywords:
                glove_dict[word] = encoder[word]

            # save_path = folder_name + '/' + str(embedding) + '_set_' +str(i) + '.npy'
            # np.save(save_path, glove_words)
            i = i+1
    else:
        keyword_sets = []
        for filename in os.listdir(folder_name):
            if filename.endswith('txt'):
                file1 = open(folder_name + filename, "r+")
                lines = file1.readlines()
                keywords = list(lines[2].strip().split(", "))
                in_text = lines[1].split()[:30]
                keyword_sets.append((' '.join(in_text), keywords))
                for word in keywords:
                    glove_dict[word] = encoder[word]

    save_path_dict = folder_name + '/dict_' + str(embedding) + '.pkl'
    with open(save_path_dict, 'wb') as f:
        pickle.dump(glove_dict, f, pickle.HIGHEST_PROTOCOL)


# if encode_articles == True:

#     for n in [4, 5, 8, 9, 10, 12, 13, 14, 15, 16]:
#         print(n)
#         file1 = open(str(os.path.dirname(os.path.abspath(__file__))) +
#                      "/data/keyword_to_articles/test_" + str(n) + ".txt", "r+")

#         lines = file1.readlines()

#         keywords = list(lines[2].strip().split(", "))
#         print(keywords)
#         glove_words = []
#         for word in keywords:
#             glove = encoder[word]
#             glove_words.append(glove)

#         save_path = str(os.path.dirname(
#             os.path.abspath(__file__))) + '/data/keyword_to_articles/test_' +str(n) + '.npy'
#         np.save(save_path, glove_words)

# 主程序：解析命令行参数，包括输入文件路径，词嵌入模型类型和任务类型，然后调用相应的创建嵌入字典的函数

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str)
    parser.add_argument('-word_embedding', type=str, default='glove',
                        choices=list(word_embedding.keys()), help='word_embedding')
    # 'key2article', 'commongen'
    parser.add_argument('-task', type=str, default=None)
    args = parser.parse_args()
    file_name = args.file
    embedding = args.word_embedding
    task = args.task

    create_enc_dict(file_name, embedding)
