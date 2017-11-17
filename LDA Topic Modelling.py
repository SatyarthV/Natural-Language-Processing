#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:39:34 2017

@author: Satyarth Vaidya
"""

import pandas as pd
import gensim
import nltk.data
from nltk.corpus import stopwords
import re

from gensim import corpora, models

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = [ '\n','/',"'—" ,' #'  ,'”',  '·',' . ','’', '‘','%','.','#','@' ,"'—",'|','«','» |','|','»','.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = "|?.!/;:()&+„.“|0-9'— ·#\r\n\r\n\r\n\xa0-\xa0\r\n\r\n/"

Yahoo_data = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Yahoo_Data.csv')
Yahoo_data = Yahoo_data.drop_duplicates(['Article'])

Handelsblatt_data = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Handelsblatt.csv')
Handelsblatt_data = Handelsblatt_data.drop('Unnamed: 0' , 1)  
Handelsblatt_data = Handelsblatt_data.drop_duplicates(['Article'])

Blog_data_1 = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Final_Blog2.csv')
Blog_data_1 = Blog_data_1.drop('Unnamed: 0' , 1)
Blog_data_1 = Blog_data_1.drop_duplicates(['Article'])


Blog_data_2 = pd.read_csv(r'/home/fractaluser/Downloads/German Data/Comments_Blog.csv')
Blog_data_2 = Blog_data_2.drop('Unnamed: 0' , 1)
Blog_data_2 = Blog_data_2.drop_duplicates(['Article'])

Taz_data = pd.read_csv('/home/fractaluser/Downloads/German Data/taz.csv')
Taz_data = Taz_data.drop('Unnamed: 0' , 1)
Taz_data = Taz_data.drop_duplicates(['Article'])

#Finance_data = pd.concat([Yahoo_data , Handelsblatt_data, Blog_data_1, Blog_data_2 , Taz_data] )
Finance_data = pd.concat([Yahoo_data , Handelsblatt_data, Blog_data_1, Taz_data] )

stop_words = stopwords.words('german') 

def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    res = res.replace('-', ' ')
    return res

stop_words_new = []
for txt in stop_words:
    stop_words_new.append(replace_umlauts(txt))

try:
    Finance_data['Article_processed'] = Finance_data['Article'].apply(lambda x: x.lower())
    Finance_data['Article_processed'] = Finance_data['Article_processed'].str.replace(r'\d+', '')
    Finance_data['Article_processed'] = Finance_data['Article_processed'].apply(replace_umlauts)
except Exception as e:
    print(str(e))
    pass
Finance_data['Tokenized'] = Finance_data['Article_processed'].apply(nltk.word_tokenize)
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in punctuation_tokens])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [ re.sub('[' + punctuation + ']', '', item) for item in x])
Finance_data['Tokenized'] = Finance_data['Tokenized'].apply(lambda x : [item for item in x if item not in stop_words_new])

texts  = Finance_data.Tokenized.tolist()
"""
stopped_tokens = Finance_data.iloc[0,4]
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

texts = [p_stemmer.stem(i) for i in stopped_tokens]
"""

## Apply 

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=6, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=3, num_words=3))
"""
Results
euro	 Euro			unternehmen	Companies		aktien	shares
prozent percent		wert	 value	         	unternehmen	Companies
mehr	 more			mal	   times		          stefan	     stefan
"""
