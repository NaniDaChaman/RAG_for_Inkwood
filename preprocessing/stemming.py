import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
import numpy as np 
#import pandas as pd 
import re

def remove_urls(q):
    #removing urls 
    url_search=re.compile(r'https?://\S+')
    return url_search.sub("",q)

#removing nonwords and non white spaces
def nws_removal(q):
    nws=re.compile(r'[^\w\s]')
    return nws.sub("",q)

#remove digits 
def digit_removal(q):
    dig=re.compile(r'\d')
    return dig.sub("",q)

#tokenisation
def token(q):
    nltk.download('punkt_tab')
    return word_tokenize(q)

#stop words removal
def stopword_removal(q):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    return [word for word in q if word not in stop_words]
    

#stemming
def stem(q):
    stemmer=PorterStemmer()
    return [stemmer.stem(word) for word in q]

def main (q):
    #
    q=q.lower()
    q=remove_urls(q)
    q=nws_removal(q)
    q=digit_removal(q)
    q=token(q)
    q=stopword_removal(q)
    q=stem(q)
    return q

if __name__=="__main__":
    q=input ("Your search query : ")
    q=main(q)
    print(f'Text after preprocessing : {q}')

#idea
#don't apply digit and lowering andstemming to proper nouns !!!
#having proper nouns in their orignal context is important
#you may wanna actually add words to proper nouns so that they make more sense 
#expand acrounyms