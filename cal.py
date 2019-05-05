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



select_word = pd.read_csv("select_words3.csv").values.tolist()
select_wordlist = []
for i in select_word:
    select_wordlist = select_wordlist + i


# テキストを引数として、形態素解析の結果、名詞のみを配列で抽出する関数を定義
t = Tokenizer()
def extract_words(text):
    tokens = t.tokenize(text)
    return [token.base_form for token in tokens 
        if token.part_of_speech.split(',')[0] in['名詞','動詞','形容詞','未定義語']]


# 全体のテキストを句点('。')で区切ったリストにし、指定のワードリスト以外の語を削除
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

# 全体のテキストを句点('。')で区切ったリストにし、指定のワードリスト以外の語を削除
def make_word_list2(text,wordlist):
    word_list = extract_words(text)

    for i in reversed(word_list):
        if i not in wordlist:
            word_list.remove(i)

    for ii in reversed(word_list):
        if len(ii) == 0:
            word_list.remove(ii)

    return word_list

"""

f = open("ramenall.txt","r",encoding = "UTF-8")
text = f.read()
f.close()

# 全体のテキストを句点('。')で区切ったリストにし、指定のワードリスト以外の語を削除
word_list = make_word_list(text,select_wordlist)
print(word_list[:10])

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
#trainings = [TaggedDocument(words = data.split(),tags = [i]) for i,data in enumerate(word_list)]


c = Doc2Vec(documents= trainings, dm = 1, window=8, min_count=1, workers=4)
c.save(".\kramenall_selectword3.model")

"""

cc = Doc2Vec.load("kramenall_selectword3.model")


with open("たかはし.txt",encoding="utf-8") as f:
    a = f.read() 
with open("どうとんぼり神座.txt",encoding="utf-8")as f:
    b = f.read()
with open("はやし田.txt",encoding="utf-8") as f:
    c = f.read()
with open("ほりうち.txt",encoding="utf-8") as f:
    d = f.read()
with open("一幻.txt",encoding="utf-8") as f:
    e = f.read()
with open("海神.txt",encoding="utf-8") as f:
    g = f.read()
with open("岐阜屋.txt",encoding="utf-8") as f:
    h = f.read()
with open("桂花.txt",encoding="utf-8") as f:
    i = f.read()
with open("広州市場.txt",encoding="utf-8") as f:
    j = f.read()
with open("竹虎.txt",encoding="utf-8") as f:
    k = f.read()
with open("中本.txt",encoding="utf-8") as f:
    l = f.read()
with open("凪.txt",encoding="utf-8") as f:
    m = f.read()
with open("八郎商店.txt",encoding="utf-8") as f:
    n = f.read()
with open("百日紅.txt",encoding="utf-8") as f:
    o = f.read()
with open("風雲児.txt",encoding="utf-8") as f:
    p = f.read()
with open("満来.txt",encoding="utf-8") as f:
    q = f.read()
with open("麺屋武蔵.txt",encoding="utf-8") as f:
    r = f.read()
with open("龍の家.txt",encoding="utf-8") as f:
    s = f.read()
with open("鈴蘭.txt",encoding="utf-8") as f:
    u = f.read()
with open("翔.txt",encoding="utf-8") as f:
    v = f.read()


blog = [a,b,c,d,e,g,h,i,j,k,l,m,n,o,p,q,r,s,u,v]


df = pd.DataFrame(index=range(20))
bbb=[]

for aaa in range(20):
    ggg_words = make_word_list2(blog[aaa],select_wordlist)
    for ccc in range(20):    
        hhh_words = make_word_list2(blog[ccc],select_wordlist)
        sim_value = cc.docvecs.similarity_unseen_docs(cc,ggg_words,hhh_words,alpha=1,min_alpha=0.0001,steps=5)
        bbb.append(sim_value)
    print(bbb)
    df[aaa] = bbb
    bbb.clear()
    print(aaa)

df.to_excel("ramen_selectword3.xlsx")
