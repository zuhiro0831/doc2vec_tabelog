#coding:utf-8

import re
from collections import OrderedDict
import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer
from datetime import datetime
import glob



if __name__== '__main__':
    
    char_filters = [UnicodeNormalizeCharFilter(), # UnicodeをNFKCで正規化
                    RegexReplaceCharFilter('\d+', '0')] # 数字を全て0に置換

    tokenizer = Tokenizer(mmap=True) # NEologdを使う場合、mmap=Trueとする

    token_filters = [POSKeepFilter(['名詞','動詞','形容詞']), # 名詞のみを取得する
                     LowerCaseFilter(), # 英字は小文字にする
                     ExtractAttributeFilter('base_form')] # 原型のみを取得する

    analyzer = Analyzer(char_filters, tokenizer, token_filters)


    komi = sorted(glob.glob("kutikomi\*.txt"))

    f = open("select_words3.txt","r",encoding = "UTF-8")
    keykey = f.read()
    keykey = keykey.replace("\ufeff","")
    list = keykey.split("\n")
    f.close()
    df = pd.DataFrame(index = list)
    cnt = 0


    for data in komi:
        texts_words = []
        with open (data,encoding = "utf-8") as k:
            d = k.read()
        
        for token in analyzer.analyze(d):
            texts_words.append(token)

        word_cnt={}
        for wcnt in list:
            word_cnt[wcnt] = 0

        for kutikomi in texts_words:
            for keyw in word_cnt.keys():
                if kutikomi == keyw:
                    word_cnt[kutikomi] += 1
        
        for wordc in list:
            df.at[wordc,cnt]=word_cnt[wordc]
        cnt +=1
            
    df.to_excel("ramenwc"+str(datetime.now().strftime("%m%d%h%m"))+".xlsx")
