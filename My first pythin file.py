
#1 pandas 프로파일링
import pandas as pd
import pandas_profiling
data = pd.read_csv('spam.csv',encoding='latin1')
data[:5]

pr=data.profile_report.to_file('./pr_report.html') #오류발생


#2 텍스트 전처리_토큰화
import tensorflow
import tensorflow.keras
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

#텐서플로우 오류
from tensorflow.keras.preprocessing.text import text_to_word_sequence
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

#띄어쓰기 토큰화
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
text="텍스트 전처리 토큰화"
print(tokenizer.tokenize(text))

#문장 토큰화
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))


#2_2 정제와 정규화
import re
text="I was wondering if anyone out there could enlighten me on this car."
shortword = re.compile(r"\w*\b\w{1,2}\b")
print(shortword.sub('',text))

#표제어 추출
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([n.lemmatize(w) for w in words])

n.lemmatize('dies','v')
n.lemmatize('watched', 'v')
n.lemmatize('has', 'v')

#어간 추출
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s=PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)

print([s.stem(w) for w in words])

#포터 어간 추출기
from nltk.stem import PorterStemmer
s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])

#랭커스터 스태머 알고리즘
from nltk.stem import LancasterStemmer
l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])

#불용어 제거
from nltk.corpus import stopwords
stopwords.words('english')[:10]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="Family is not an important thing. It's everything."
stop_words=set(stopwords.words("english"))


word_tokes=word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
print(result)

#한국어 불용어 제거 불용어(의미없는 단어), 이 리스트는 사전에 정의
example="고기를 아무렇게나 구우려고 하면 안돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 떄는 중요한게 있지."
stop_words="아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨데 이럴정도로 하면 아니거든"

stop_words=stop_words.split(' ')
word_tokens=word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
print(result)

##정규 표현식
#기초
import re
r=re.compile("a.c")
r.search("kkk")
r.search("abc")

import re
r=re.compile("ab?c")
r.search("abc")
r.search("ac")

r=re.compile("ab*c")
r.search("a")
r.search("ac")
r.search("abc")
r.search("abbbbc")

import re
r=re.compile("ab+c")
r.search("ac")
r.search("abc")

r=re.compile("^a")
r.search("bbc")
r.search("ab")

r=re.compile("ab{2}c")
r.search("ac")
r.search("abc")
r.search("abbc")

r=re.compile("ab{2,8}c")
r.search("ac")
r.search("ac")
r.search("abc")

r.search("abbbbc")

import re
r=re.compile("[abc]")
r.search("a")

r.search("aaaaa")
r.search("baaac")

import re
r=re.compile("[a-z]")
r.search("AAA")

r.search("aBC")
r.search("111")

import re
r=re.compile("[^abc]")
r.search("a")
r.search("ab")
r.search("b")
r.search("d")

r.search("1")

# re.match와 re.search의 차이
r=re.compile("ab.")
r.search("kkkabc")
r.match("kkkabc")
r.match("abckkk")

#re.split()
text="사과 딸기 수박 메론 바나나"
re.split(" ",text)

text='''사과
딸기
수박
메론
바나나'''
re.split("\n",text)

text="사과+딸기+수박+메론+바나나"
re.split("\+",text)

#re.findall()
text="이름: 김철수 전화번호 : 010 - 1234 - 1234 나이 : 30 성별 : 남"
re.findall("\d+",text)

import re
text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
re.sub('[^a-zA-Z]',' ',text)


text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""

re.split('\s+', text)
re.findall('\d+',text)

re.findall('[A-Z]',text)

re.findall('[A-Z]{4}',text)

re.findall('[A-Z][a-z]+',text)

letters_only=re.sub('[^a-zA-Z]'," ",text)

#정규 표현식을 이용한 토큰화
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

#6장 토픽 모델링
import numpy as np
A=np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])
np.shape(A)

U, s, VT = np.linalg.svd(A, full_matrices = True)
print(U.round(2))
np.shape(U)

print(s.round(2))
np.shape(s)

S=np.zeros((4,9))
S[:4,:4]=np.diag(s)
print(S.round(2))
np.shape(S)

print(VT.round(2))
np.shape(VT)

A_prime=np.dot(np.dot(U,S), VT)
print(A)
print(A_prime.round(2))


from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
len(documents)
