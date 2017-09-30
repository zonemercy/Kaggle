# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:23:59 2017

@author: mariosm
"""
import os
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy import sparse as ssp
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import distance
from sklearn.model_selection import KFold
stop_words = stopwords.words('english')
    
#stops = set(stopwords.words("english"))
stops = set(["http","www","img","border","home","body","a","about","above","after","again","against","all","am","an",
"and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't",
"cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
"herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought",
"our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such",
"than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're",
"they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
"weren't","what","what's","when","when's""where","where's","which","while","who","who's","whom","why","why's","with","won't","would",
"wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves" ])
porter = PorterStemmer()
snowball = SnowballStemmer('english')

weights={}

def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):    
    zsparse=csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))    
    print(" indptr lenth %d" % (len(indptr)))
    
    f=open(filename,"w")
    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if ytarget!=None:
             f.write(str(ytarget[b]) + deli1)     
             
        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))                    
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))  
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:    
            print(" row : %d " % (counter_row))    
    f.close()  
    


# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=5000.0, min_count=2.0):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


def word_shares(row,wei,stop):
    
 
		q1 = set(str(row['question1']).lower().split())
		q1words = q1.difference(stop)
		if len(q1words) == 0:
			return '0:0:0:0:0'

		q2 = set(str(row['question2']).lower().split())
		q2words = q2.difference(stop)
		if len(q2words) == 0:
			return '0:0:0:0:0'

		q1stops = q1.intersection(stop)
		q2stops = q2.intersection(stop)

		shared_words = q1words.intersection(q2words)
		#print(len(shared_words))
		shared_weights = [wei.get(w, 0) for w in shared_words]
		total_weights = [wei.get(w, 0) for w in q1words] + [wei.get(w, 0) for w in q2words]
        
		R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
		R2 = float(len(shared_words)) / (float(len(q1words)) + float(len(q2words))) #count share
		R31 = float(len(q1stops)) / float(len(q1words)) #stops in q1
		R32 = float(len(q2stops)) / float(len(q2words)) #stops in q2
		return '{}:{}:{}:{}:{}'.format(R1, R2, float(len(shared_words)), R31, R32)

def stem_str(x,stemmer=SnowballStemmer('english')):
        x = text.re.sub("[^a-zA-Z0-9]"," ", x)
        x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
        x = " ".join(x.split())
        return x
    
def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)

def str_abs_diff_len(str1, str2):
    return abs(len(str1)-len(str2))

def str_len(str1):
    return len(str(str1))

def char_len(str1):
    str1_list = set(str(str1).replace(' ',''))
    return len(str1_list)

def word_len(str1):
    str1_list = str1.split(' ')
    return len(str1_list)

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return R

def str_jaccard(str1, str2):


    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res

# shortest alignment
def str_levenshtein_1(str1, str2):


    #str1_list = str1.split(' ')
    #str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=1)
    return res

# longest alignment
def str_levenshtein_2(str1, str2):

    #str1_list = str1.split(' ')
    #str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=2)
    return res

def str_sorensen(str1, str2):

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res



def main():

    path="" # set your input folder here
   
  
    ######## from here on starts qqgeogor example from (https://www.kaggle.com/)#######
    #https://github.com/qqgeogor/kaggle_quora_benchmark
    
    
     ################### generate_stem .py################## 
    seed = 1024
    np.random.seed(seed)
    path = os.path.expanduser("~") + "/data/quora/"
    #re load to avoid errors. 
    
    train = pd.read_csv(path+"train.csv")
    test = pd.read_csv(path+"test.csv")

    print('Generate porter')
    train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),snowball))
    test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),snowball))
    
    train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),snowball))
    test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),snowball))
    
    train.to_csv(path+'train_porter.csv')
    test.to_csv(path+'test_porter.csv')
    

    ###################### generate_interaction.py ################    
    
    train = pd.read_csv(path+"train_porter.csv")
    test = pd.read_csv(path+"test_porter.csv")
    test['is_duplicated']=[-1]*test.shape[0]
    
    print('Generate intersection')
    train_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
    test_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
    pd.to_pickle(train_interaction,path+"train_interaction.pkl")
    pd.to_pickle(test_interaction,path+"test_interaction.pkl")
    
    print('Generate porter intersection')
    train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
    test_porter_interaction = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
    
    pd.to_pickle(train_porter_interaction,path+"train_porter_interaction.pkl")
    pd.to_pickle(test_porter_interaction,path+"test_porter_interaction.pkl")  
    
    ###################### generate_tfidf.py ################  

        
    ft = ['question1','question2','question1_porter','question2_porter']
    train = pd.read_csv(path+"train_porter.csv")[ft]
    test = pd.read_csv(path+"test_porter.csv")[ft]
    # test['is_duplicated']=[-1]*test.shape[0]
    
    data_all = pd.concat([train,test])
    print data_all
    
    max_features = None
    ngram_range = (1,2)
    min_df = 3
    print('Generate tfidf')
    feats= ['question1','question2']
    vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)
    
    corpus = []
    for f in feats:
        data_all[f] = data_all[f].astype(str)
        corpus+=data_all[f].values.tolist()
    
    vect_orig.fit(corpus)
    
    for f in feats:
        tfidfs = vect_orig.transform(data_all[f].values.tolist())
        train_tfidf = tfidfs[:train.shape[0]]
        test_tfidf = tfidfs[train.shape[0]:]
        pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
        pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)
    
    
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
        train_tfidf = tfidfs[:train.shape[0]]
        test_tfidf = tfidfs[train.shape[0]:]
        pd.to_pickle(train_tfidf,path+'train_%s_tfidf.pkl'%f)
        pd.to_pickle(test_tfidf,path+'test_%s_tfidf.pkl'%f)    
        
        
    ##################### generate_len.py #########################
    
    train = pd.read_csv(path+"train_porter.csv").astype(str)
    test = pd.read_csv(path+"test_porter.csv").astype(str)
    
    print('Generate len')
    feats = []
    
    train['abs_diff_len'] = train.apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
    test['abs_diff_len']= test.apply(lambda x:str_abs_diff_len(x['question1'],x['question2']),axis=1)
    feats.append('abs_diff_len')
    
    train['R']=train.apply(word_match_share, axis=1, raw=True)
    test['R']=test.apply(word_match_share, axis=1, raw=True)
    feats.append('R')
    
    train['common_words'] = train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    test['common_words'] = test.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    feats.append('common_words')
    
    for c in ['question1','question2']:
        train['%s_char_len'%c] = train[c].apply(lambda x:char_len(x))
        test['%s_char_len'%c] = test[c].apply(lambda x:char_len(x))
        feats.append('%s_char_len'%c)
    
        train['%s_str_len'%c] = train[c].apply(lambda x:str_len(x))
        test['%s_str_len'%c] = test[c].apply(lambda x:str_len(x))
        feats.append('%s_str_len'%c)
        
        train['%s_word_len'%c] = train[c].apply(lambda x:word_len(x))
        test['%s_word_len'%c] = test[c].apply(lambda x:word_len(x))
        feats.append('%s_word_len'%c)
    

    pd.to_pickle(train[feats].values,path+"train_len.pkl")
    pd.to_pickle(test[feats].values,path+"test_len.pkl")       
    
    #########################generate_distance.py #################

    train = pd.read_csv(path+"train_porter.csv")
    test = pd.read_csv(path+"test_porter.csv")
    test['is_duplicated']=[-1]*test.shape[0]
    
    data_all = pd.concat([train,test])    
    
    print('Generate jaccard')
    train_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
    test_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
    pd.to_pickle(train_jaccard,path+"train_jaccard.pkl")
    pd.to_pickle(test_jaccard,path+"test_jaccard.pkl")
    
    print('Generate porter jaccard')
    train_porter_jaccard = train.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
    test_porter_jaccard = test.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
    
    pd.to_pickle(train_porter_jaccard,path+"train_porter_jaccard.pkl")
    pd.to_pickle(test_porter_jaccard,path+"test_porter_jaccard.pkl")  

    # path=""
    ###################  generate_svm_format_tfidf.py ################# 
    train = pd.read_csv(path+"train_porter.csv")    
    
    train_question1_tfidf = pd.read_pickle(path+'train_question1_tfidf.pkl')[:]
    test_question1_tfidf = pd.read_pickle(path+'test_question1_tfidf.pkl')[:]

    
    train_question2_tfidf = pd.read_pickle(path+'train_question2_tfidf.pkl')[:]
    test_question2_tfidf = pd.read_pickle(path+'test_question2_tfidf.pkl')[:]
 

    #train_question1_porter_tfidf = pd.read_pickle(path+'train_question1_porter_tfidf.pkl')[:]
    #test_question1_porter_tfidf = pd.read_pickle(path+'test_question1_porter_tfidf.pkl')[:]
    
    #train_question2_porter_tfidf = pd.read_pickle(path+'train_question2_porter_tfidf.pkl')[:]
    #test_question2_porter_tfidf = pd.read_pickle(path+'test_question2_porter_tfidf.pkl')[:]
    
    
    train_interaction = pd.read_pickle(path+'train_interaction.pkl')[:].reshape(-1,1)
    test_interaction = pd.read_pickle(path+'test_interaction.pkl')[:].reshape(-1,1)

    train_interaction=np.nan_to_num(train_interaction)
    test_interaction=np.nan_to_num(test_interaction)      

    
    train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')[:].reshape(-1,1)
    test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')[:].reshape(-1,1)


    train_porter_interaction=np.nan_to_num(train_porter_interaction)
    test_porter_interaction=np.nan_to_num(test_porter_interaction)
    
    
    train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')[:].reshape(-1,1)
    test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')[:].reshape(-1,1)


    train_jaccard=np.nan_to_num(train_jaccard)
    test_jaccard=np.nan_to_num(test_jaccard)
    
    train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')[:].reshape(-1,1)
    test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')[:].reshape(-1,1)


    train_jaccard=np.nan_to_num(train_jaccard)
    test_porter_jaccard=np.nan_to_num(test_porter_jaccard)
    
    train_len = pd.read_pickle(path+"train_len.pkl")
    test_len = pd.read_pickle(path+"test_len.pkl")
    
    train_len=np.nan_to_num(train_len)
    test_len=np.nan_to_num(test_len) 
    

    scaler = MinMaxScaler()
    scaler.fit(np.vstack([train_len,test_len]))
    train_len = scaler.transform(train_len)
    test_len =scaler.transform(test_len)
 
    
    
    X = ssp.hstack([
        train_question1_tfidf,
        train_question2_tfidf,
        train_interaction,
        train_porter_interaction,
        train_jaccard,
        train_porter_jaccard,
        train_len
        ]).tocsr()
    
    
    y = train['is_duplicate'].values[:]
    
    X_t = ssp.hstack([
        test_question1_tfidf,
        test_question2_tfidf,
        test_interaction,
        test_porter_interaction,
        test_jaccard,
        test_porter_jaccard,
        test_len
        ]).tocsr()
    
    
    print X.shape
    print X_t.shape
    


    skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
    for ind_tr, ind_te in skf:
        X_train = X[ind_tr]
        X_test = X[ind_te]

        y_train = y[ind_tr]
        y_test = y[ind_te]
        break

    # dump_svmlight_file(X,y,PATH+"X_tfidf.svm")
    # del X
    # dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),PATH+"X_t_tfidf.svm")
    # del X_t



    def oversample(X_ot,y,p=0.175):
        pos_ot = X_ot[y==1]
        neg_ot = X_ot[y==0]
        #p = 0.165
        scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
        while scale > 1:
            neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
            scale -=1
        neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
        ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
        y=np.zeros(ot.shape[0])
        y[:pos_ot.shape[0]]=1.0
        print y.mean()
        return ot,y

    X_train,y_train = oversample(X_train.tocsr(),y_train,p=0.175)
    X_test,y_test = oversample(X_test.tocsr(),y_test,p=0.175)

    X_train,y_train = shuffle(X_train,y_train,random_state=seed)  



    # fromsparsetofile(path + "X_tfidf.svm", X, deli1=" ", deli2=":",ytarget=y)
    # del X
    # fromsparsetofile(path + "X_t_tfidf.svm", X_t, deli1=" ", deli2=":",ytarget=None)
    # del X_t
    # fromsparsetofile(path + "X_train_tfidf.svm", X_train, deli1=" ", deli2=":",ytarget=y_train)
    # fromsparsetofile(path + "X_test_tfidf.svm", X_test, deli1=" ", deli2=":",ytarget=y_test)

    dump_svmlight_file(X,y,path+"X_tfidf.svm")
    dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),path+"X_t_tfidf.svm")
    del X_t
    del X
    dump_svmlight_file(X_train,y_train,path+"X_train_tfidf.svm")
    dump_svmlight_file(X_test,y_test,path+"X_test_tfidf.svm")



    print ("done!")      
                     
if __name__=="__main__":
    main()
  