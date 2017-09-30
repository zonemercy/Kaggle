import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
SEED = 2048
np.random.seed(SEED)
PATH = os.path.expanduser("~") + "/data/quora/"

train = pd.read_csv(PATH + "train.csv")#, nrows=5000)
test = pd.read_csv(PATH + "test.csv")#, nrows=5000)

def stem_str(x,stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

porter = PorterStemmer()
snowball = SnowballStemmer('english')

print ('Generate porter')
train['question1_porter'] = train['question1'].astype(str).apply(lambda x: stem_str(x.lower(),snowball))
test['question1_porter'] = test['question1'].astype(str).apply(lambda x: stem_str(x.lower(),snowball))
train['question2_porter'] = train['question2'].astype(str).apply(lambda x: stem_str(x.lower(),snowball))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x: stem_str(x.lower(),snowball))

train.to_csv(PATH+'train_porter.csv')
test.to_csv(PATH+'test_porter.csv')
