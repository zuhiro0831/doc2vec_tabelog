#coding:utf-8

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
from collections import OrderedDict
from janome.tokenizer import Tokenizer
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
from janome.tokenfilter import POSKeepFilter, LowerCaseFilter, ExtractAttributeFilter
from janome.analyzer import Analyzer



# テキストを引数として、形態素解析の結果、名詞のみを配列で抽出する関数を定義
def extract_words(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens 
        if token.part_of_speech.split(',')[0] in['名詞','動詞','形容詞','未定義語']]


# 全体のテキストを句点('。')で区切ったリスト(入れ子)にし、指定のワードリスト以外の語を削除
def make_word_list(text,wordlist):
    sentences = text.split('。')
    word_list = [extract_words(sentence) for sentence in sentences]

    for i in word_list:
        for li in reversed(i):
            if li not in wordlist:
                i.remove(li)

    for ii in reversed(word_list):
        if len(ii) == 0:
            word_list.remove(ii)
    
    return word_list



if __name__== '__main__':
    
    select_word = pd.read_csv("select_words3.csv").values.tolist()
    select_wordlist = []
    for i in select_word:
        select_wordlist = select_wordlist + i
        

    f = open("ramenall.txt","r",encoding = "UTF-8")
    text = f.read()
    f.close()

    word_list = make_word_list(text,select_wordlist)

    trainings=[]
    cnt = 1
    for word in word_list:
        bbb = TaggedDocument(words = word,tags=[cnt])
        trainings.append(bbb)
        cnt += 1

    print(trainings[:10])


    #空白で単語を区切り、改行で文書を区切っているテキストデータ

    #１文書ずつ、単語に分割してリストに入れていく[([単語1,単語2,単語3],文書id),...]こんなイメージ
    #words：文書に含まれる単語のリスト（単語の重複あり）
    # tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）

    c = Doc2Vec(documents= trainings, dm = 1, window=8, min_count=1, workers=4)
    c.save(".\kramenall_selectword3.model")
