#!/usr/bin/env python
# -*- coding:utf-8 -*-
# by pppihf
import re
import jieba
from jieba import analyse
import settings
from ltp import LTP
import os
import pandas as pd
from tqdm import tqdm
from LAC import LAC
import stanza


class ChinesePreprocessor:
    def __init__(self, stopwords_path):
        """
        :param stopwords_path: 停用词文件路径
        """
        self.stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.link = re.compile(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
            r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
        # 装载LAC分词模型
        self.lac = LAC(mode='seg')
        # self.extract = analyse.extract_tags

    def word_cut(self, text: str):
        """
        将传入的list进行清洗和分词
        :param text: 待分词字符串
        :return: 分词后的文档字符串
        """
        # 去除回车符和换行符
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 去除http链接
        text = re.sub(pattern=self.link, repl=' ', string=text)
        # 去除特殊字符
        text = re.sub(pattern=self.SPECIAL_SYMBOL_RE, repl=' ', string=text)
        ## jieba分词
        # segments = jieba.cut(text)  # 默认精确模式切割词语
        # LAC分词，单个样本输入，输入为Unicode编码的字符串
        segments = self.lac.run(text)
        # 去停用词
        words = ""  # 返回值是字符串
        for w in segments:
            # if len(w) < 2:  # 去除单个字符
            #     continue
            if w.isdigit():  # 去除完全为数字的字符串
                continue
            if w not in self.stopwords:  # 去除停用词
                words += w
                words += " "
        # print(words+"***")
        return words

    # def get_keywords(self, content: str, topK=5):
    #     """
    #     使用TFIDF算法获取关键词，最多获取5个TFIDF值超过0.2的关键词
    #     :param content: 原始文本
    #     :param topK: 关键词数量
    #     :return: 关键词列表
    #     """
    #     keywords = []
    #     try:
    #         tags = self.extract(content, topK, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    #         i = 0
    #         for v, n in tags:
    #             if i > 4:
    #                 break
    #             if n < 0.2:
    #                 break
    #             keywords.append(v)
    #             i = i+1
    #     except Exception as e:
    #         pass
    #     return keywords


class NamedEntity:
    def __init__(self, stopwords_path=settings.STOPWORDS_DATA, user_dict=settings.USER_DICT_DATA):
        self.stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
        self.SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')  # 删除特殊符号
        self.link = re.compile(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
            r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
        # # 加载LTP
        # self.ltp = LTP(settings.LTP_MODEL)  # 默认加载Small模型
        # # user_dict.txt 是词典文件， max_window是最大前向分词窗口
        # self.ltp.init_dict(path=user_dict, max_window=4)
        # # 加载stanza
        # self.zh_nlp = stanza.Pipeline('zh', processors='tokenize,ner')
        # 加载LAC
        self.lac = LAC(mode='lac')

    def entity_recognition(self, text):
        """
        命名实体识别
        :param text: 原始文本
        :return: 从原始文本中抽取的命名实体
        """
        # # 使用LTP
        # seg, hidden = self.ltp.seg(text)   # 分词
        # ner = self.ltp.ner(hidden)
        # entity = []
        # for tag, start, end in ner[0]:
        #     entity.append(seg[0][start:end+1][0])

        # # 使用stanza
        # doc = self.zh_nlp(text)
        # entity = []
        # for sent in doc.sentences:
        #     # print("NER: " + ' '.join(f'{ent.text}/{ent.type}' for ent in sent.ents), flush=True)
        #     for ent in sent.ents:
        #         entity.append(ent.text)

        # 使用LAC，单个样本输入，批量输入会占用很多资源
        # 清理文本
        text = text.replace('\n', ' ').replace('\r', ' ')  # 去除回车符和换行符
        text = re.sub(pattern=self.link, repl=' ', string=text)  # 去除http链接
        text = re.sub(pattern=self.SPECIAL_SYMBOL_RE, repl=' ', string=text)  # 去除特殊字符
        # 进行词性标注
        seg, pos = self.lac.run(text)
        # 清理分词结果
        words = [w.strip() for w in seg if not w.isdigit()]  # 去除纯数字
        words = [w for w in words if len(w) >= 2]  # 去除单字
        words = [w for w in words if w not in self.stopwords]  # 去除停用词
        words = [w for w in words if re.match(pattern=r' +', string=w) is None]  # 去除纯空格
        # 提取实体
        entities = [w for (w, p) in zip(seg, pos) if p in ['PER', 'LOC', 'ORG', 'TIME']]
        # 合并分词和实体
        words.extend(entities)
        return words


def wordcut(data, text_column, label_column, infile=None, outfile=None):
    """
    分词
    :param data:  待分词的数据，dataframe格式
    :param text_column:  待分词的列名
    :param label_column:  标签的列名
    :param infile:  待分词的文件，csv格式
    :param outfile:  保存的分词结果，csv文件
    :return: 分词结果，列表
    """
    if infile is not None:
        # 读取原始数据
        data = pd.read_csv(infile, encoding='utf-8')
    # content = list(data['content'])
    # 开启分词
    Cut = ChinesePreprocessor(stopwords_path=settings.STOPWORDS_DATA)
    # 使用LAC
    new_data = data.copy().loc[:, [label_column, text_column]]
    new_data.columns = ['flag', 'content']
    tqdm.pandas(desc="pandas bar")
    new_data['words'] = new_data.progress_apply(lambda x: Cut.word_cut(x['content']), axis=1)
    new_data = new_data.loc[:, ['flag', 'words', 'content']]
    # 不使用LAC
    # corpus = []
    # result = []
    # for i in range(data.shape[0]):
    #     L = []
    #     L.append(data[label_column][i])
    #     word = Cut.word_cut(data[text_column][i])
    #     corpus.append(word)
    #     L.append(word)
    #     result.append(L)
    # new_data = pd.DataFrame(data=result, columns=['flag', 'words', 'content'])
    # 保存为dataframe
    new_data.dropna(inplace=True)
    corpus = new_data.content.to_list()
    if outfile is not None:
        new_data.to_csv(outfile, index=False, encoding='utf-8')
    print("Wordcut is done.")
    return new_data, corpus


def wordcut_ner(data, text_column, label_column, infile=None, outfile=None):
    """
    分词，并把提取的实体加到分词结果中
    :param data:  待分词的数据，dataframe格式
    :param text_column:  待分词的列名
    :param label_column:  标签的列名
    :param infile:  待分词的文件，csv格式
    :param outfile:  保存的分词结果，csv文件
    :return: 分词结果，列表
    """
    if infile is not None:
        # 读取原始数据
        data = pd.read_csv(infile, encoding='utf-8')
    # # 开启分词
    # Cut = ChinesePreprocessor(stopwords_path=settings.STOPWORDS_DATA)
    # 命名实体识别
    NE = NamedEntity(user_dict=settings.USER_DICT_DATA, stopwords_path=settings.STOPWORDS_DATA)
    # 使用LAC
    new_data = data.copy().loc[:, [label_column, text_column]]
    new_data.columns = ['flag', 'content']
    tqdm.pandas(desc="pandas bar")
    new_data['words'] = new_data.progress_apply(lambda x: ' '.join(NE.entity_recognition(x['content'])), axis=1)
    new_data = new_data.loc[:, ['flag', 'words', 'content']]
    # 不使用LAC
    # corpus = []
    # result = []
    # for i in tqdm(range(data.shape[0])):
    #     L = []
    #     L.append(data[label_column][i])
    #     word = Cut.word_cut(data[text_column][i])
    #     # print(word)
    #     entity = NE.entity_recognition(data[text_column][i])
    #     # print(entity)
    #     if len(entity) >= 1:
    #         word = word + ' '.join(entity)
    #     # print(word)
    #     L.append(word)
    #     corpus.append(word)
    #     result.append(L)
    # new_data = pd.DataFrame(data=result, columns=['flag', 'words', 'content'])
    # 保存为dataframe
    new_data.dropna(inplace=True)
    corpus = new_data.content.to_list()
    if outfile is not None:
        new_data.to_csv(outfile, index=False, encoding='utf-8')
    print("Wordcut and NER is done.")
    return new_data, corpus

# if __name__ == '__main__':
#     filename = os.path.join(settings.DATA_DIR, 'labeled_data.csv')
#     outputfile = settings.DATA_DIR + 'new_cut_data.csv'
#     NER = True
#     if NER:
#         # 分词+命名实体识别
#         result = wordcut_ner(infile=filename, text_column='title', label_column='flag', outfile=outputfile)
#         # result = tfidf_ner_dbscan(infile=filename, outfile=outputfile)
#     else:
#         # 分词
#         result = wordcut(infile=filename, text_column='title', label_column='flag', outfile=outputfile)
