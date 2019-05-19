#coding:utf-8

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer
import glob


# テキストを引数として、形態素解析の結果、名詞のみを配列で抽出する関数を定義
def extract_words(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens 
        if token.part_of_speech.split(',')[0] in['名詞','動詞','形容詞','未定義語']]


# 全体のテキストを単語ごとに区切ったリストにし、指定のワードリスト以外の語を削除
def make_word_list2(text,wordlist):
    word_list = extract_words(text)

    for i in reversed(word_list):
        if i not in wordlist:
            word_list.remove(i)

    for ii in reversed(word_list):
        if len(ii) == 0:
            word_list.remove(ii)

    return word_list


if __name__=='__main__':
    
    select_word = pd.read_csv("select_words3.csv").values.tolist()
    select_wordlist = []
    for i in select_word:
        select_wordlist = select_wordlist + i

    cc = Doc2Vec.load("kramenall_selectword3.model")

    #同ディレクトリの中に、店舗それぞれの口コミデータを全て入れたフォルダ「kutikomi」を作成しておく
    komi = sorted(glob.glob("kutikomi\*.txt"))

    #リストにそれぞれの店舗の口コミデータを処理したものを格納
    ddata=[]
    for d in komi:
        with open(d,encoding="utf-8") as f:
            r = f.read()
        ddata.append(make_word_list2(r,select_wordlist))

    #店舗同士の口コミ文書の類似度を計算→DataFrameへ格納
    df = pd.DataFrame(index=range(20))
    bbb=[]
    for aaa in range(20):
        for ccc in range(20):
            sim_value = cc.docvecs.similarity_unseen_docs(cc,ddata[aaa],ddata[ccc],alpha=1,min_alpha=0.0001,steps=5)
            bbb.append(sim_value)
        print(bbb)
        df[aaa] = bbb
        bbb.clear()
        print(aaa)

    df.to_excel("ramen_selectword.xlsx")
