import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 2048
np.random.seed(SEED)
PATH = os.path.expanduser("~") + "/data/quora/"

cols = ['question1','question2','question1_porter','question2_porter']
train = pd.read_csv(PATH + "train_porter.csv")[cols]#, nrows=5000)[cols]
test = pd.read_csv(PATH + "test_porter.csv")[cols]#, nrows=5000)[cols]

len_train = train.shape[0]

data_all = pd.concat([train,test])
print data_all.head()

max_features = None
ngram_range = (1,2)
min_df = 3

print('Generate_tfidf')
feats = ['question1','question2']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range,min_df=min_df)

corpus =[]
for f in feats:
	data_all[f] = data_all[f].astype(str)
	corpus+=data_all[f].values.tolist()

vect_orig.fit(corpus)

for f in feats:
	tfidfs = vect_orig.transform(data_all[f].values.tolist())
	train_tfidf = tfidfs[:len_train]
	test_tfidf = tfidfs[len_train:]
	pd.to_pickle(train_tfidf, PATH+'train_%s_tfidf.pkl'%f)
	pd.to_pickle(test_tfidf, PATH+'test_%s_tfidf.pkl'%f)

print('Generate porter tfidf')
feats= ['question1_porter','question2_porter']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:len_train]
    test_tfidf = tfidfs[len_train:]
    pd.to_pickle(train_tfidf,PATH+'train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,PATH+'test_%s_tfidf.pkl'%f)