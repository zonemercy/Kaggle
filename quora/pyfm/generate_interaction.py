import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity,pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
SEED = 2048
np.random.seed(SEED)
PATH = os.path.expanduser("~") + "/data/quora/"

train = pd.read_csv(PATH + "train_porter.csv")#, nrows=5000)
test = pd.read_csv(PATH + "test_porter.csv")#, nrows=5000)
test['is_duplicated'] = [-1]*test.shape[0]

len_train = train.shape[0]

data_all = pd.concat([train,test])

def calc_set_intersection(obj,target):
	a = set(obj.split())
	b = set(target.split())
	return (len(a.intersection(b))*1.0) / (len(a)*1.0)

print('Generate intersection')
train_interaction = train.astype(str).apply(lambda x: calc_set_intersection(x['question1'],x['question2']),axis=1)
test_interaction = test.astype(str).apply(lambda x: calc_set_intersection(x['question1'],x['question2']),axis=1)
pd.to_pickle(train_interaction,PATH+"train_interaction.pkl")
pd.to_pickle(test_interaction,PATH+"test_interaction.pkl")

print('Generate porter intersection')
train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
test_porter_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
pd.to_pickle(train_porter_interaction, PATH+"train_porter_interaction.pkl")
pd.to_pickle(test_porter_interaction, PATH+"test_porter_interaction.pkl")