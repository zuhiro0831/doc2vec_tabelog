from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer
import numpy as np
import pandas as pd


# 頻出する単語を抽出
def get_selectwords(texts_words):
    word_count = {}
    for word in texts_words:
        if not word in word_count:
            word_count[word] = 0
        word_count[word] += 1
    return {k for k,v in word_count.items() if v >= len(texts_words) * 0.0001}
    # 0.01%以上の文書に含まれる単語を取得


if __name__ == '__main__':

    f = open("ramenall.txt","r",encoding = "UTF-8")
    r = f.read()
    f.close()

    char_filters = [UnicodeNormalizeCharFilter(), # UnicodeをNFKCで正規化
                    RegexReplaceCharFilter('\d+', '0')] # 数字を全て0に置換
    tokenizer = Tokenizer(mmap=False)
    token_filters = [POSKeepFilter(['名詞','動詞','形容詞',"未定義語"]), # 名詞、動詞、形容詞、未定義語のみを取得する
                     LowerCaseFilter(), # 英字は小文字にする
                     ExtractAttributeFilter('base_form')] # 原型のみを取得する
    analyzer = Analyzer(char_filters, tokenizer, token_filters)

    texts_words=[]
    for token in analyzer.analyze(r):
        texts_words.append(token)

    selectwords = get_selectwords(texts_words)

    swlist = list(selectwords)
    df = pd.DataFrame()
    df[0] = swlist
    df.to_excel("pic_words.xlsx")
