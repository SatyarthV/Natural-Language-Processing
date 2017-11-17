#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:18:03 2017

@author: Satyarth Vaidya
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

dataframe = pd.read_fwf('/home/Downloads/linear_regression_demo-master/brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values,y_values)
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()

import numpy as np
np.random.random_sample((3,2))-5

from gensim import models
model = word2vec(sentences, size=100, window=5, min_count=5, workers=4)


model = gensim.models.KeyedVectors.load_word2vec_format(args.model.strip(), binary=True)





src = '/home/firsttest.text'
model = gensim.models.KeyedVectors.load_word2vec_format('/home/Downloads/german (2).model', binary=True)
out_lis = []

def test_mostsimilar(model,src, label='most similar', topn=10):
    num_lines = sum(1 for line in open(src))
    num_questions = 0
    num_right = 0
    num_topn = 0
    # get questions
    with open(src) as f:
        questions = f.readlines()
        questions = [x.strip() for x in questions]
    # test each question
    for question in questions:
        words = question.split()
        # check if all words exist in vocabulary
        if all(x in model.index2word for x in words):
            num_questions += 1
            #bestmatches = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=topn)
            bestmatches = model.most_similar(positive= 'Hallo' , topn=topn)
            print(bestmatches)
            # best match
            """
            if words[3] in bestmatches[0]:
                num_right += 1
            # topn match
            for topmatches in bestmatches[:topn]:
                if words[3] in topmatches:
                    num_topn += 1
                    break
    # calculate result
    correct_matches = round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    topn_matches = round(num_topn/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.
    
    """



