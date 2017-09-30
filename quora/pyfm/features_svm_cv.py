import os
import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
SEED = 2048
np.random.seed(SEED)
PATH = os.path.expanduser("~") + "/data/quora/"
train = pd.read_csv(PATH+"train_porter.csv")

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


X = ssp.hstack([
    train_question1_tfidf,
    train_question2_tfidf,
    train_interaction,
    train_porter_interaction,
    train_jaccard,
    train_porter_jaccard,
    train_len,
    ]).tocsr()


y = train['is_duplicate'].values[:]

X_t = ssp.hstack([
    test_question1_tfidf,
    test_question2_tfidf,
    test_interaction,
    test_porter_interaction,
    test_jaccard,
    test_porter_jaccard,
    test_len,
    ]).tocsr()


print X.shape
print X_t.shape

skf = KFold(n_splits=5, shuffle=True, random_state=SEED).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

dump_svmlight_file(X,y,PATH+"X_tfidf.svm")
del X
dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),PATH+"X_t_tfidf.svm")
del X_t

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

X_train,y_train = shuffle(X_train,y_train,random_state=SEED)

dump_svmlight_file(X_train,y_train,PATH+"X_train_tfidf.svm")
dump_svmlight_file(X_test,y_test,PATH+"X_test_tfidf.svm")
