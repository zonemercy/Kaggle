from __future__ import division
import pandas as pd
import numpy as np
import random, os, gc
import config
from scipy import sparse as ssp
from sklearn.utils import resample,shuffle
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, f_classif
from sklearn import preprocessing
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
RAW_PATH=config.RAW_PATH
FEAT_PATH =config.FEAT_PATH

train = pd.read_csv(RAW_PATH+'train.csv')
train.drop(train.index[[config.ab_dup_test]], inplace=True)
train.reset_index(drop=True, inplace=True)
train_y = train.is_duplicate.values

feat_df = ['feat_ab.csv','feature_vect_lemmer.csv','feat_158_stpf.csv']

df = pd.read_csv(FEAT_PATH+'magic_feature.csv')
del df['question1'], df['question2'], df['id']
print 'feat_mag {}'.format(df.shape)

def remove_col(train):
    list1=['question1','question2','id','is_duplicate']
    for i in list1:
        if i in list(train.columns):
            del train[i]
    return train

for f in feat_df:
    df1 = pd.read_csv(FEAT_PATH+f)
    df1 = remove_col(df1)
    df = pd.concat([df, df1],axis=1)
    del df1
    gc.collect()
    print f, df.shape


feature_base_close_porter = pd.read_csv(FEAT_PATH+'feature_base_close_porter.csv')
del feature_base_close_porter['question1'], feature_base_close_porter['question2'], feature_base_close_porter['is_duplicate']
print 'feature_base_close_porter {}'.format(feature_base_close_porter.shape)

df = pd.concat([df, feature_base_close_porter], axis=1)
print 'df: {}'.format(df.shape)
del feature_base_close_porter
gc.collect()

del_feat = ['q1_hash','q2_hash','q_hash_pos','q_hash_pos_1','q1_change','q2_change']
del_feat.extend(['q_change_pair','q1_q2_change_max'])
del_feat.extend(['euclidean_distance', 'jaccard_distance','RMSE_distance'])
del_feat.extend(['freq_diff', 'q1_q2_intersect_ratio'])
del_feat.extend(list(tr_corr[abs(tr_corr['is_duplicate'])<0.01].index))

print df.shape
for i in list(df.columns):
    if i in del_feat:
        del df[i]
# df = df[use_feat]
print df.shape


########### select k best features #############
train = df[df['is_duplicate']!=-1].copy()
train =train.replace([np.inf, -np.inf], np.nan).dropna()

full_feat = list(train.columns)
full_feat.remove('is_duplicate')
min_max_scaler = preprocessing.MinMaxScaler()
train[full_feat] = min_max_scaler.fit_transform(train[full_feat])

selector = SelectKBest(chi2, k=200)
selector.fit(train[full_feat], train['is_duplicate'])
idxs_selected = selector.get_support(indices=True)
columns_selected = train[full_feat].columns[idxs_selected]
print columns_selected
del train
gc.collect()

df = df[list(columns_selected)+['is_duplicate']]


def oversample(X_ot,y,p=0.173):
    raw_num = X_ot.shape[0]
    print "RAW shape: {} | Mean rate: {}".format(X_ot.shape[0], y.mean())
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]
    #p = 0.165
    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = np.vstack([neg_ot, neg_ot])
        scale -=1
    neg_ot = np.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]])
    ot = np.vstack([pos_ot, neg_ot])
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    print "Oversample: {} | Mean rate: {}".format(ot.shape[0],y.mean())
    return ot,y


test = df[df['is_duplicate']==-1].copy()
del test['is_duplicate']
train = df[df['is_duplicate']!=-1].copy()
del train['is_duplicate']
del df
gc.collect()
print train.shape, test.shape

############### drop absolute duplicate rows #################
train.drop(train.index[[config.ab_dup_test]], inplace=True)
train.reset_index(drop=True, inplace=True)


embeddings_index = {}
f = open(config.RAW_PATH+'glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()



tk = text.Tokenizer(nb_words=200000)
max_len = 140


tk.fit_on_texts(list(test.question1.values) + list(test.question2.values.astype(str)))
x1_test = tk.texts_to_sequences(test.question1.values)
x1_test = sequence.pad_sequences(x1_test, maxlen=max_len)
x2_test = tk.texts_to_sequences(test.question2.values.astype(str))
x2_test = sequence.pad_sequences(x2_test, maxlen=max_len)


kf = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)
for train_idx, valid_idx in kf.split(train, y=train_y):
    X_train, X_test, y_train, y_test = train.loc[train_idx,:], train.loc[valid_idx,:], train_y[train_idx],train_y[valid_idx]
    print X_train.shape, y_train.shape
    X_train,y_train = oversample(X_train,y_train,p=0.1742)
    X_test,y_test = oversample(X_test,y_test,p=0.1742)
    X_train,y_train = shuffle(X_train,y_train,random_state=42)  
    print X_train.shape, y_train.shape



    tk = text.Tokenizer(nb_words=200000)
    max_len = 140

    tk.fit_on_texts(list(train.question1.values) + list(train.question2.values.astype(str)))
    x1 = tk.texts_to_sequences(X_train.question1.values)
    x1 = sequence.pad_sequences(x1, maxlen=max_len)
    x2 = tk.texts_to_sequences(X_train.question2.values.astype(str))
    x2 = sequence.pad_sequences(x2, maxlen=max_len)

    word_index = tk.word_index


    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    max_features = 200000
    filter_length = 5
    nb_filter = 64
    pool_length = 4

    model = Sequential()
    print('Build model...')

    model1 = Sequential()
    model1.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))

    model1.add(TimeDistributed(Dense(300, activation='relu')))
    model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

    model2 = Sequential()
    model2.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))

    model2.add(TimeDistributed(Dense(300, activation='relu')))
    model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

    model3 = Sequential()
    model3.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))
    model3.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model3.add(Dropout(0.2))

    model3.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))

    model3.add(GlobalMaxPooling1D())
    model3.add(Dropout(0.2))

    model3.add(Dense(300))
    model3.add(Dropout(0.2))
    model3.add(BatchNormalization())

    model4 = Sequential()
    model4.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=40,
                         trainable=False))
    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model4.add(Dropout(0.2))

    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))

    model4.add(GlobalMaxPooling1D())
    model4.add(Dropout(0.2))

    model4.add(Dense(300))
    model4.add(Dropout(0.2))
    model4.add(BatchNormalization())
    model5 = Sequential()
    model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

    model6 = Sequential()
    model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(300))
    merged_model.add(PReLU())
    merged_model.add(Dropout(0.2))
    merged_model.add(BatchNormalization())

    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))

    merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

    merged_model.fit( [x1_train, x2_train, x1_train, x2_train, x1_train, x2_train], y=y_train, \
                    validation_data=([x1_valid, x2_valid, x1_valid, x2_valid, x1_valid, x2_valid], y_labels), \
                    batch_size=384, nb_epoch=200, verbose=1, shuffle=True, callbacks=[checkpoint] )