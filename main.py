#!/usr/bin/python
# -*- encoding:utf-8 -*-
# by pppihf
import settings
import argparse
import json
import pandas as pd
from Modules.text_process import wordcut, wordcut_ner
from Modules.tf_idf import feature_vector
from Modules.static_dbscan import my_db, show_db


def data_process( text_column, label_column, do_ner, data=None, infile=None, outfile=None):
    # 对文本进行处理
    # 返回值：处理后的Dataframe（包含标签，处理后的文本，原文），处理后的文本列表
    if do_ner:
        # 分词+命名实体识别
        result, corpus = wordcut_ner(data, text_column, label_column, infile, outfile)
    else:
        # 分词
        result, corpus = wordcut(data, text_column, label_column, infile, outfile)
    return result, corpus


def generate_feature(dimension, data=None, file=None):
    if file is not None:
        data = pd.read_csv(file, sep=',', encoding='utf-8')
    # 数据特征化
    data = data.words.values.tolist()
    trainingData = feature_vector(data, dimension=dimension)
    return trainingData


def cluster(eps, min_sample, metric, corpus_embeddings, corpus=None, label_data=None, corpus_file=None,
            label_file=None):
    # 聚类
    db = my_db(eps, min_sample, metric, corpus_embeddings)
    show_db(label_data, db, corpus, corpus_embeddings, corpus_file, label_file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='Input', type=str, default=None, help='要读取的csv文件')
    parser.add_argument('-t', '--text_column', dest='TextColumn', type=str, default=None, help='文本在csv文件中的列名')
    parser.add_argument('-l', '--label_column', dest='LabelColumn', type=str, default=None, help='标签在csv文件中的列名')
    parser.add_argument('-o', '--output', dest='Output', type=str, default=None, help='输出路径')
    args = parser.parse_args()  # 字典的方式接收参数
    print(args)

    # 测试输入格式为json字符串
    data = pd.read_csv(args.Input, encoding="utf-8")
    data = data.to_json(orient='records')
    # data = json.loads(data)  # 得到list
    data = pd.read_json(data, orient='records')  # 得到dataframe
    print(data)

    result, corpus = data_process(text_column=args.TextColumn,
                                  label_column=args.LabelColumn,
                                  do_ner=True,
                                  data=data,
                                  # infile=args.Input,
                                  outfile=settings.PROCESSED_DATA,
                                  )
    corpus_embeddings = generate_feature(dimension=8,
                                         # data=result,
                                         file=settings.PROCESSED_DATA,
                                         )
    cluster(eps=0.085,
            min_sample=8,
            metric='cosine',  # ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’,‘hamming’, ‘jaccard’, etc.
            corpus_embeddings=corpus_embeddings,
            # corpus=corpus,
            # label_data=result,
            corpus_file=settings.PROCESSED_DATA,
            label_file=settings.PROCESSED_DATA,
            )
