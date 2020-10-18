#!/usr/bin/python
# -*- encoding:utf-8 -*-
# by pppihf
import os
################
# Project Configuration
################

# 项目路径
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# 数据存放路径
DATA_DIR = os.path.join(BASE_PATH, 'Data')

# 停用词存放路径
STOPWORDS_DIR = os.path.join(DATA_DIR, 'stopwords')
# 使用的停用词文件
STOPWORDS_DATA = os.path.join(STOPWORDS_DIR, '呆萌的停用词表.txt')

# 使用的用户自定义字典文件
USER_DICT_DATA = os.path.join(DATA_DIR, 'user_dict.txt')

# 文本处理后的保存路径
PROCESSED_DATA = os.path.join(DATA_DIR, 'processed_data.csv')

# LTP模型文件
LTP_MODEL = r'F:\Python\LTP_Models\base.tgz'

# Path to plot
PLOT_DIR = os.path.join(BASE_PATH, 'plot')
