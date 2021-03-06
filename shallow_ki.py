# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# Clone the entire repo.
!git clone -l -s https://github.com/VishalPallagani/rqe-nlp-mediqa2019.git RQE
# %cd RQE
!ls

!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_bc5cdr_md-0.2.0.tar.gz 
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_bionlp13cg_md-0.2.0.tar.gz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import nltk
import re
import scispacy
import spacy
from spacy import displacy
from collections import Counter
#import en_core_web_sm
import en_ner_bc5cdr_md
#import en_core_sci_sm
#import en_core_sci_md
import en_ner_bionlp13cg_md
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from collections import OrderedDict
from pprint import pprint
import urllib, json

def parse_XML(xml_file, df_cols): 
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
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

def get_entities(model,document):
    """  
    This function displays word entities

    Parameters: 
         model(module): A pretrained model from spaCy(https://spacy.io/models) or ScispaCy(https://allenai.github.io/scispacy/)
         document(str): Document to be processed

    Returns: Image rendering and list of named/unnamed word entities and entity labels 
     """
    nlp = model.load()
    doc = nlp(document)
    #displacy_image = displacy.render(doc, jupyter=True,style='ent')
    entity_and_label = list(set([(X.text, X.label_) for X in doc.ents]))
    return entity_and_label

train['outcome'] = np.where(train['value'] == 'true', 1, 0)
val['outcome'] = np.where(val['value'] == 'true', 1, 0)
test['outcome'] = np.where(test['value'] == 'true', 1, 0)
train.head()

y_train = train['outcome']
X_train = train.drop(columns = ['pid', 'type', 'value', 'outcome'])

y_test = test['outcome']
X_test = test.drop(columns = ['pid', 'type', 'value', 'outcome'])

X_train

x = X_test['chq'].values.tolist()
x = [i.replace("\n","") for i in x]
X_test['chq'] = x

X_train.shape

final = []
for i in X_test['chq'].values.tolist():
  final.append(i)

len(final)

disease = []
for i in final:
  b = get_entities(en_ner_bc5cdr_md,i)
  top = []
  for j in b:
    if(j[1]=='DISEASE'):
      top.append(j[0])
  disease.append(top)

terms = []
for i in disease:
  top = []
  if(len(i)>0):
    for j in i:
      a = j.lower().split(" ")
      if(len(a)==1):
        try:
          url = "https://clinicaltrials.gov/api/query/full_studies?expr=" + a[0] + "&min_rnk=1&max_rnk=1&fmt=json"
          response = urllib.request.urlopen(url)
          data = json.loads(response.read())
          sub = data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf']
        except:
          #terms.append(top)
          pass
        for k in sub:
          if(k['ConditionBrowseLeafRelevance']=='high'):
            top.append(k['ConditionBrowseLeafName'])
      elif(len(a)>1):
        substr = ""
        for m in a:
          substr = substr + m + "+"
        try:
          url = "https://clinicaltrials.gov/api/query/full_studies?expr=" + substr + "&min_rnk=1&max_rnk=1&fmt=json"
          response = urllib.request.urlopen(url)
          data = json.loads(response.read())
          sub = data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf']
        except:
          #terms.append(top)
          pass
        for k in sub:
          if(k['ConditionBrowseLeafRelevance']=='high'):
            top.append(k['ConditionBrowseLeafName'])
  terms.append(top)

df = pkl.load(open('saved.pkl','rb'))

print(disease[6])
print(df[6])

new = []
for i,j in zip(final, terms):
  sub = ""
  for k in j:
    sub = sub + k +" "
  st = ""
  st = st + i + " " + sub
  new.append(st)

final[6]

new[6]

newdf = pd.DataFrame(data = None, columns=['chq','faq','label'])

y_test

newdf['chq'] = new
newdf['faq'] = X_test['faq'].values.tolist()
newdf['label'] = y_test.values

newdf

newdf.to_csv('Train_NLI.csv')

newdf = pd.read_csv('Train_NLI.csv')

import pickle as pkl
pkl.dump(disease,open('disease.pkl','wb'))

pkl.dump(terms, open('clinicaltrials.pkl','wb'))

newdf['label'] = y_train.values

X_train.head()

print(disease[-1])
print(terms[-1])

url = 'https://clinicaltrials.gov/api/query/full_studies?expr=kartagener+syndrome&min_rnk=1&max_rnk=1&fmt=json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

a = data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf']

for i in a:
  if(i['ConditionBrowseLeafRelevance']=='high'):
    print(i['ConditionBrowseLeafName'])

if(data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf'][2]['ConditionBrowseLeafRelevance']=='high'):
  print(data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf'][2]['ConditionBrowseLeafName'])











test_doc = final[1]

b = get_entities(en_ner_bc5cdr_md,test_doc)

b















!pip install -U spacy
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz

import scispacy
import spacy
import en_core_sci_sm
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

nlp = spacy.load("en_core_sci_sm")

text = '''I am suffering from Kartagener's syndrome and wanted information from you or from Dr. [NAME]. for this syndrome. (About fertility) and if possible other symptoms. Thank you.'''

doc = nlp(text)

print(doc.ents)

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
#Print the Abbreviation and it's definition
print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
      print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")

from scispacy.umls_linking import UmlsEntityLinker

nlp = spacy.load("en_core_sci_sm")
linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(linker)
# Each entity is linked to UMLS with a score
# (currently just char-3gram matching).
for umls_ent in entity._.umls_ents:
          print(linker.umls.cui_to_entity[umls_ent[0]])



!pip install spacy scispacy https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

import scispacy
import spacy
import en_core_sci_lg
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

nlp = en_core_sci_lg.load()

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
linker = UmlsEntityLinker(resolve_abbreviations=True, max_entities_per_mention=1)
nlp.add_pipe(linker)

text = "I am suffering from Kartagener's syndrome and wanted information from you or from Dr. [NAME]. for this syndrome. (About fertility) and if possible other symptoms. Thank you."

res = nlp(text)

!pip install textacy
import textacy

res = nlp(text)

res

abbrs = res._.abbreviations
for abrv in abbrs:
  print(abrv, " - ", abrv._.long_form)



entity = res.ents
print(entity)

for i in range(len(entity)):
  for umls_ent in entity[i]._.umls_ents:
    # print(linker.umls.cui_to_entity[umls_ent[0]])
    print(linker.umls.cui_to_entity[umls_ent[0]][1], " - ", linker.umls.cui_to_entity[umls_ent[0]][2][-1])
    print(linker.umls.cui_to_entity[umls_ent[0]][4],"\n")

from medacy.model.model import Model

model = Model.load_external('medacy_model_clinical_notes')
annotation = model.predict("The patient was prescribed 1 capsule of Advil for 5 days.")
print(annotation)



























import urllib, json
url = 'https://clinicaltrials.gov/api/query/full_studies?expr=Kartagener+Syndrome&min_rnk=1&max_rnk=1&fmt=json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

if(data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf'][2]['ConditionBrowseLeafRelevance']=='high'):
  print(data['FullStudiesResponse']['FullStudies'][0]['Study']['DerivedSection']['ConditionBrowseModule']['ConditionBrowseLeafList']['ConditionBrowseLeaf'][2]['ConditionBrowseLeafName'])

data







!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_bc5cdr_md-0.2.0.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_bionlp13cg_md-0.2.0.tar.gz

import scispacy
import spacy
from spacy import displacy
from collections import Counter
#import en_core_web_sm
import en_ner_bc5cdr_md
#import en_core_sci_sm
#import en_core_sci_md
import en_ner_bionlp13cg_md
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from collections import OrderedDict
from pprint import pprint

test_doc = "I am suffering from Kartagener's syndrome and wanted information from you or from Dr. [NAME]. for this syndrome. (About fertility) and if possible other symptoms. Thank you. What is the cause of Polymicrogyria ? What is primary ciliary dyskinesia ? What causes Dehydration ? Can being overweight cause IBS ?How much does it cost to surgically remove a papilloma ?"

def display_entities(model,document):
    """  
    This function displays word entities

    Parameters: 
         model(module): A pretrained model from spaCy(https://spacy.io/models) or ScispaCy(https://allenai.github.io/scispacy/)
         document(str): Document to be processed

    Returns: Image rendering and list of named/unnamed word entities and entity labels 
     """
    nlp = model.load()
    doc = nlp(document)
    displacy_image = displacy.render(doc, jupyter=True,style='ent')
    entity_and_label = list(set([(X.text, X.label_) for X in doc.ents]))
    return entity_and_label

display_entities(en_ner_bionlp13cg_md,test_doc)

test, test1 = display_entities(en_ner_bc5cdr_md,test_doc)

model = en_ner_bc5cdr_md

nlp = model.load()
doc = nlp(test_doc)
displacy_image = displacy.render(doc, jupyter=True,style='ent')
entity_and_label = set([(X.text, X.label_) for X in doc.ents])

print(entity_and_label)

test

req = []
for i in test1:
  if(i[1] == 'DISEASE'):
    req.append(i[0])
req



#BERT

dftrain = pd.read_csv('Train_NLI.csv')
dftest = pd.read_csv('Test_NLI.csv')

X_train = dftrain.drop(columns = ['label'])
y_train = dftrain['label']
X_test = dftest.drop(columns = ['label'])
y_test = dftest['label']

!git clone -b master https://github.com/charles9n/bert-sklearn
!cd bert-sklearn; pip install .
import os
os.chdir("bert-sklearn")

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

model = BertClassifier(max_seq_length=64, train_batch_size=16)
model.num_mlp_layers = 3
model

"""```
# This is formatted as code
```



1.   Shallow KI + BERT -- 49.56
2.   BERT -- 47.9
"""

scores_BERT = []; 
for seed in [4, 27, 33]:
    model.random_state = seed
    model.fit(X_train, y_train)
    scores_BERT.append(model.score(X_test, y_test))

scores_BERT

sum(scores_BERT)/3

y_pred = model.predict(X_test)
print((X_test[y_pred == y_test].shape[0])/X_test.shape[0])

import csv
count = 0
with open('correct_bert.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred[i] == y_test[i]):
      #print(X_test.iloc[i]
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred[i]])

count = 0
with open('incorrect_bert.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred[i] != y_test[i]):
      #print(X_test.iloc[i]
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred[i]])

# BioBERT
model_biobert = BertClassifier(max_seq_length=64, train_batch_size=16)
model_biobert.num_mlp_layers = 3
model_biobert.bert_model = 'biobert-base-cased' 
model_biobert

"""1.   Shallow KI + BioBERT -- 51.15
2.   BioBERT -- 50.144
"""

scores_bioBERT = []; 
for seed in [4, 27, 33]:
    model_biobert.random_state = seed
    model_biobert.fit(X_train, y_train)
    scores_bioBERT.append(model_biobert.score(X_test, y_test))

sum(scores_bioBERT)/3

y_pred_biobert = model_biobert.predict(X_test)
print((X_test[y_pred_biobert == y_test].shape[0])/X_test.shape[0])

count = 0
with open('incorrect_biobert.csv', 'w') as f:
  writer = csv.writer (f)
  writer.writerow(['chq','faq','target', 'predicted'])
  for i in range(230):  
    if(y_pred_biobert[i] != y_test[i]):
      #print(X_test.iloc[i]
      writer.writerow([X_test['chq'][i], X_test['faq'][i], y_test[i], y_pred_biobert[i]])

#Bag of Words Model

import pandas as pd
df = pd.read_csv('NLI_data.csv')

df['question'] = df['chq'] + df['faq']

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import tensorflow
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import re

from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def normalize_text(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    s = re.sub('\s+',' ',s)
    
    return s

df['question'] = [normalize_text(s) for s in df['question']]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['question'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

nb = MultinomialNB()
nb.fit(x_train, y_train)

print(nb.score(x_test, y_test))

y_pred = nb.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
