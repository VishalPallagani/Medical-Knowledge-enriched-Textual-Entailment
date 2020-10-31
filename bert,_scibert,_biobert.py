# -*- coding: utf-8 -*-
"""BERT, SciBERT, BioBERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lwHCYZuvQxF1bLaTpslJ4x_suMnQLyiP
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone -l -s https://github.com/VishalPallagani/rqe-nlp-mediqa2019.git RQE
# %cd RQE
!ls

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import nltk
import re

def parse_XML(xml_file, df_cols): 
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        res.append(node.attrib.get(df_cols[1]))
        res.append(node.attrib.get(df_cols[2]))
        for el in df_cols[3:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df

train = parse_XML('/content/RQE/data/RQE_Train_8588_AMIA2016.xml', ['pid', 'type', 'value', 'chq', 'faq'])
val = parse_XML('/content/RQE/data/RQE_Test_302_pairs_AMIA2016.xml', ['pid', 'type', 'value', 'chq', 'faq'])
test = parse_XML('/content/RQE/data/MEDIQA2019-Task2-RQE-TestSet-wLabels.xml', ['pid', 'type', 'value', 'chq', 'faq'])
train.head()

train['outcome'] = np.where(train['value'] == 'true', 1, 0)
val['outcome'] = np.where(val['value'] == 'true', 1, 0)
test['outcome'] = np.where(test['value'] == 'true', 1, 0)
train.head()

"""## MLP"""

!git clone -b master https://github.com/charles9n/bert-sklearn
!cd bert-sklearn; pip install .
import os
os.chdir("bert-sklearn")
print(os.listdir())

import os
import math
import random
import csv
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import statistics as stats

from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model

# BERT
model = BertClassifier(max_seq_length=64, epochs = 8, train_batch_size=16)
model.num_mlp_layers = 3
model

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

y_train = train['outcome']
X_train = train.drop(columns = ['outcome','Unnamed: 0'])

y_test = test['outcome']
X_test = test.drop(columns = ['outcome','Unnamed: 0'])

X_train

y_train = train['outcome']
X_train = train.drop(columns = ['pid', 'type', 'value', 'outcome'])

y_test = test['outcome']
X_test = test.drop(columns = ['pid', 'type', 'value', 'outcome'])

X_train

X_train.to_csv('XTrain.csv')

y_train.to_csv('YTrain.csv')

X_test.to_csv('XTest.csv')

y_test.to_csv('YTest.csv')

scores_BERT = []; 
for seed in [4, 27, 33]:
    model.random_state = seed
    model.fit(X_train, y_train)
    scores_BERT.append(model.score(X_test, y_test))

scores_BERT

y_pred = model.predict(X_test)
print((X_test[y_pred == y_test].shape[0])/X_test.shape[0])

sum(scores_BERT)/3

model_scibert = BertClassifier(max_seq_length=64, train_batch_size=16)
model_scibert.num_mlp_layers = 3
model_scibert.bert_model = 'biobert-v1.0-pubmed-pmc-base-cased'
model_scibert

scores_sciBERT = []; 
for seed in [4, 27, 33]:
    model_scibert.random_state = seed
    model_scibert.fit(X_train, y_train)
    scores_sciBERT.append(model_scibert.score(X_test, y_test))

sum(scores_sciBERT)/3

from sklearn.metrics import precision_recall_fscore_support

y_pred_scibert = model_scibert.predict(X_test)
print((X_test[y_pred_scibert == y_test].shape[0])/X_test.shape[0])

precision_recall_fscore_support(y_test, y_pred_scibert, average='weighted')

import csv
count = 0
with open('incorrect.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred_scibert[i] != y_test[i]):
      #print(X_test.iloc[i]
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred_scibert[i]])

model_scibert = BertClassifier(max_seq_length=64, train_batch_size=16)
model_scibert.num_mlp_layers = 3
model_scibert.bert_model = 'scibert-scivocab-uncased'
model_scibert

scores_sciBERT = []; 
for seed in [4, 27, 33]:
    model_scibert.random_state = seed
    model_scibert.fit(X_train, y_train)
    scores_sciBERT.append(model_scibert.score(X_test, y_test))

scores_sciBERT

y_pred_scibert = model_scibert.predict(X_test)
print((X_test[y_pred_scibert == y_test].shape[0])/X_test.shape[0])

X_test.head(2)

import csv
count = 0
with open('correct.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred_scibert[i] == y_test[i]):
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred_scibert[i]])

pwd

sum(scores_sciBERT)/3

target_names = ['negative', 'positive']
print(classification_report(y_test, y_pred_scibert, target_names=target_names))

savefile = 'scibert_model.bin'
model_scibert.save(savefile)

"""BioBERT"""

model_biobert = BertClassifier(max_seq_length=64, train_batch_size=16)
model_biobert.num_mlp_layers = 3
model_biobert.bert_model = 'biobert-base-cased' 
model_biobert

scores_bioBERT = []; 
for seed in [4, 27, 33]:
    model_biobert.random_state = seed
    model_biobert.fit(X_train, y_train)
    scores_bioBERT.append(model_biobert.score(X_test, y_test))

scores_bioBERT

y_pred_biobert = model_biobert.predict(X_test)
print((X_test[y_pred_biobert == y_test].shape[0])/X_test.shape[0])

sum(scores_bioBERT)/3

test[y_pred_biobert == y_test]

test[y_pred_biobert != y_test]

import csv
count = 0
with open('incorrect_biobert.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred_biobert[i] != y_test[i]):
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred_biobert[i]])