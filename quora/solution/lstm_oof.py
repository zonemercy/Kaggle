########################################
## import packages
########################################
import os, gc
import re
import csv
import codecs
import numpy as np
import pandas as pd
from string import punctuation
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import KFold, StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import config
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = config.RAW_PATH
EMBEDDING_FILE = BASE_DIR + 'glove.840B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train_nn.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

print STAMP

########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE)
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

texts_1 = [] 
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3], remove_stopwords=False))
        texts_2.append(text_to_wordlist(values[4], remove_stopwords=False))
#         texts_1.append(values[3])
#         texts_2.append(values[4])
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE) as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(text_to_wordlist(values[1], remove_stopwords=False))
        test_texts_2.append(text_to_wordlist(values[2], remove_stopwords=False))
#         test_texts_1.append(values[1])
#         test_texts_2.append(values[2])
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)

########################################
## generate leaky features
########################################

leak_df = pd.read_csv(config.FEAT_PATH+'magic_feature.csv')
del leak_df['question1'], leak_df['question2'], leak_df['id']
print 'feat_mag {}'.format(leak_df.shape)
leak_df.drop(leak_df.index[[config.ab_dup_test]], inplace=True)
leak_df.reset_index(drop=True, inplace=True)
print 'feat_mag {}'.format(leak_df.shape)


leak_feat = ['q1_freq', 'q2_freq', 'freq_diff', 'q1_q2_intersect', 'q1_q2_wm_ratio', \
             'q1_pr', 'q2_pr', 'q1_kcores', 'q2_kcores']
leaks = leak_df[leak_df['is_duplicate']!=-1][leak_feat]
test_leaks = leak_df[leak_df['is_duplicate']==-1][leak_feat]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)
del leak_df
gc.collect()

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


########################################
## sample train/validation data
########################################

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

test = pd.read_csv(config.RAW_PATH+'test.csv')
test_array = np.zeros(test.shape[0],dtype='float32')
oof_array = []
del test
gc.collect()

for oof_cv in [0,1,2,3,4]:  ##########  CONTROL OOF CV FOLD~~~~~~~~~~~~~

    train = pd.read_csv(config.RAW_PATH+'train_nn.csv')
    train_y = train['is_duplicate'].values
    kf = StratifiedKFold(n_splits=5, random_state=1988, shuffle=True)
    for train_idx, valid_idx in list(kf.split(train, y=train_y))[oof_cv:]:
        print train_idx, valid_idx
        idx_train, idx_val = train_idx, valid_idx
        break

    data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
    data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
    leaks_train = np.vstack((leaks[idx_train], leaks[idx_train]))
    labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

    data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
    data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
    leaks_val = np.vstack((leaks[idx_val], leaks[idx_val]))
    labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

    data_1_oof = data_1[idx_val]
    data_2_oof = data_2[idx_val]
    leaks_oof = leaks[idx_val]
    labels_oof = labels[idx_val]
    print data_1_oof.shape, data_2_oof.shape, leaks_oof.shape, labels_oof.shape

    data_1_oof,labels_oof = oversample(data_1_oof,labels[idx_val],p=0.1742)
    data_2_oof,labels_oof = oversample(data_2_oof,labels[idx_val],p=0.1742)
    leaks_oof,labels_oof = oversample(leaks_oof,labels[idx_val],p=0.1742)

    print data_1_oof.shape, data_2_oof.shape, leaks_oof.shape, labels_oof.shape


    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344

    ########################################
    ## define the model structure
    ########################################
    embedding_layer = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    leaks_input = Input(shape=(leaks.shape[1],))
    leaks_dense = Dense(num_dense//2, activation=act)(leaks_input)

    merged = concatenate([x1, y1, leaks_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## add class weight
    ########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()
    print(STAMP)

    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1_train, data_2_train, leaks_train], labels_train, \
            validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val), \
            epochs=200, batch_size=2048, shuffle=True, \
            class_weight=class_weight, callbacks=[early_stopping, model_checkpoint], verbose=0)

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    print bst_val_score
    ########################################
    ## make the submission
    ########################################
    print('Start making the submission before fine-tuning')

    preds_oof = model.predict([data_1_oof, data_2_oof, leaks_oof], batch_size=8192, verbose=1)
    preds_oof += model.predict([data_2_oof, data_1_oof, leaks_oof], batch_size=8192, verbose=1)
    preds_oof /= 2
    oof_array.extend(list(preds_oof.ravel()))


    preds = model.predict([test_data_1, test_data_2, test_leaks], batch_size=8192, verbose=1)
    preds += model.predict([test_data_2, test_data_1, test_leaks], batch_size=8192, verbose=1)
    preds /= 2
    test_array = test_array + preds.ravel()

    print oof_cv




test_array = test_array / 5.0

submission = pd.DataFrame({'lstm2':oof_array})
submission.to_csv(config.SUB_PATH+STAMP+'_oof'+'.csv', index=False)
submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':test_array})
submission.to_csv(config.SUB_PATH+STAMP+'_test'+'.csv', index=False)
