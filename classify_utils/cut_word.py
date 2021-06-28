#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:52:02 2021

@author: nijiahui
"""

import jieba
import pickle
from collections import Counter

def jieba_cut_word(content,cut_all=False):
    """使用jieba处理分词 - 分词默认使用精确模式 , or cut_all=True全模式
        paras:
            content:需分词的文本
        return:
            cuted_list:包含词组的list
    """
    seg_list = jieba.cut(content, cut_all=False)  
    cuted_list = list(seg_list)
    
    # if not cut_all:
    #     print("精准模式: " + "/ ".join(seg_list)) 
    
    return cuted_list


def ltp_cut_word(content):
    """使用ltp处理分词
        paras:
            content:需分词的文本
        return:
            cuted_list:包含词组的list
    """
    pass
    
    # return cuted_list

def make_dict(title_list,cutword_method):
    """ 生成所有数据的词组字典集合
        paras:
            title_list:所有文本字典
            cutword_method:分词方法
        return:
            所有数据的词组字典列表
    """
    # 定义字典集合
    word_list = list()
    
    for title_content in title_list:
        if cutword_method == "jieba":
            cuted_list = jieba_cut_word(title_content,cut_all=False)
        elif cutword_method == "ltp":
            cuted_list = ltp_cut_word(title_content)
        else:
            raise ValueError("未发现该截词方法%s"%cutword_method)
                
        word_list.extend(cuted_list)
        
    return word_list
    
def generate_dict(word_list):
    """生成字典"""
    # 生成字典-频率
    word_dict_count = Counter(word_list)
    
    # 生成列表-按频率排序
    word_list_count = sorted(word_dict_count.items(), key=lambda x: x[1], reverse=True)
    
    # 生成字典-按频率排序
    word_dict = {item[0]:index for index,item in enumerate(word_list_count)}
    
    # 生成字典 - 无序
    # word_dict = {value:key for key,value in enumerate(list(set(word_list)))}
    return word_dict


def save_dict(data_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

def generate_word_dict(title_list,cutword_method="jieba",generate_pkl=True):
    # 定义返回数据类型
    word_dict = dict()
    
    # 生成所有数据分词列表
    word_list = make_dict(title_list,cutword_method)
    
    # 生成字典
    word_dict = generate_dict(word_list)
    
    # 生成pkl文件
    save_dict(word_dict, "./word_dict.pkl")
    
    return word_dict








