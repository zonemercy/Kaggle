# coding: utf-8
import networkx as nx
from math import sqrt
from datetime import datetime
from csv import DictReader

from config import path

def mean(x):
    return sum(x)/float(len(x))

def median(x):
    len_2 = len(x)/2
    return x[len_2]

def std(x):
    mean_x = mean(x)
    s = 0.0
    for xx in x:
        s+=(xx-mean_x)**2
    s/=len(x)
    s = sqrt(s)
    return s




def prepare_graph(paths):
    G = nx.Graph()

    idf_dict = dict()
    for path in paths:
        print(path)
        c = 0
        start = datetime.now()

        for t, row in enumerate(DictReader(open(path), delimiter=',')):
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1'])
            q2 = str(row['question2'])
            G.add_edge(q1,q2)
            c+=1
    end = datetime.now()
    print('times:',end-start)
    return G



def prepare_clique_stats(path,out,idf_dict):

    print path
    c = 0
    start = datetime.now()
    header =    ['min_max_q',
                'max_max_q',
                'min_min_q',
                'max_min_q',
                'min_median_q',
                'max_median_q',
                'min_mean_q',
                'max_mean_q',
                'min_std_q',
                'max_std_q',]
    header = ','.join(header)

    with open(out, 'w') as outfile:
        outfile.write('%s\n'%header)
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            if c%100000==0:
                print 'finished',c
            q1 = str(row['question1'])
            q2 = str(row['question2'])

            q1_idf = idf_dict.get(q1,[0.0])
            q2_idf = idf_dict.get(q2,[0.0])

            max_q1 = max(q1_idf)
            min_q1 = min(q1_idf)
            median_q1 = median(q1_idf)
            mean_q1 = mean(q1_idf)
            std_q1 = std(q1_idf)

            max_q2 = max(q2_idf)
            min_q2 = min(q2_idf)
            median_q2 = median(q2_idf)
            mean_q2 = mean(q2_idf)
            std_q2 = std(q2_idf)


            min_max_q = max([max_q1,max_q2])
            max_max_q = max([max_q1,max_q2])

            min_min_q = min([min_q1,min_q2])
            max_min_q = max([min_q1,min_q2])

            min_median_q = min([median_q1,median_q2])
            max_median_q = max([median_q1,median_q2])

            min_mean_q = min([mean_q1,mean_q2])
            max_mean_q = max([mean_q1,mean_q2])

            min_std_q = min([std_q1,std_q2])
            max_std_q = max([std_q1,std_q2])


            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                min_max_q,
                max_max_q,
                min_min_q,
                max_min_q,
                min_median_q,
                max_median_q,
                min_mean_q,
                max_mean_q,
                min_std_q,
                max_std_q,
                ))

            c+=1
            end = datetime.now()


    print 'times:',end-start

G = prepare_graph([path+'train.csv'])
cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        c = max_clique.get(item,[])
        c.append(len_clique)
        max_clique[item]=c  
    cnt+=1

prepare_clique_stats(path+'train.csv',path+'train_clique_stats_sep.csv',max_clique)

G = prepare_graph([path+'train.csv',path+'test.csv'])
cnt=0
max_clique=dict()
for clique in nx.find_cliques(G):
    if cnt%100000==0:
        print("deal cnts %d" %cnt)
    len_clique=len(clique)
    for item in clique:
        c = max_clique.get(item,[])
        c.append(len_clique)
        max_clique[item]=c  
    cnt+=1

prepare_clique_stats(path+'test.csv',path+'test_clique_stats_sep.csv',max_clique)