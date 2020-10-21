#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author ZhangLiang
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import settings
import jieba
# 由于textrank4zh 使用的是结巴分词，为了在短文本更好的提取关键词
# 加载用户自定义的词语，使得分词更准确。
jieba.load_userdict(settings.USER_DICT_DATA)


class TEXT_SUMMARY:
    def __init__(self, path):
        """
        初始化函数接口，加载停用词表
        :param path: 停用词表存储路径
        """
        self.tr4w = TextRank4Keyword(stop_words_file=path)
        self.tr4s = TextRank4Sentence(stop_words_file=path)

    def get_key(self, cluster: list, words_num=-1, sentences_num=-1):
        """
        根据传入的聚类结果提取关键词，关键句
        :param cluster: 某一聚类类别的原始文本;格式类似['我喜欢吃烧烤。','新时代四大名著有哪些？']
        :param words_num: 从文本中提取的关键词个数；默认-1，不执行操作
        :param sentences_num: 从文本中提取的关键句个数；默认-1，不执行操作
        :return: 返回关键词列表，关键句列表
        """
        # 将list聚合成一个文本
        # 注意：要求text必须是utf8编码的bytes或者str对象
        text = ''
        for c in cluster:
            if c[-1] not in ['！', '。', '？', "；", "!"]:
                c = c+'。'  # 用句号将不同的句子进行隔
            text = text+c
        keywords = []
        key_sentences = []
        if words_num > 0:
            # 分析文本
            self.tr4w.analyze(text=text, lower=True, window=2)
            # 提取关键词,关键词最少包含2个汉字
            for item in self.tr4w.get_keywords(num=words_num, word_min_len=2):
                keywords.append(item.word)
        if sentences_num > 0:
            # 分析文本
            self.tr4s.analyze(text=text, lower=True, source='all_filters')
            # 提取关键句;这里设定为关键句长度不能小于10
            for item in self.tr4s.get_key_sentences(num=sentences_num, sentence_min_len=10):
                key_sentences.append(item.sentence)
        return keywords, key_sentences



