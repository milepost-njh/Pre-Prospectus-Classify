#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:52:02 2021

@author: nijiahui
"""
import pandas as pd

from classify_utils import cut_word


def get_int_data(x,word_dict):
    word_list = cut_word.jieba_cut_word(x)
    return [word_dict.get(word,0) for word in word_list]
    

def word2int(df_data,word_dict):
    title_data = df_data["text"]
    label_data = df_data["label"]
    
    word2int_list = list(title_data.apply(lambda x:get_int_data(x,word_dict)))
    # word2int_df = pd.DataFrame(word2int_list)
    return word2int_list