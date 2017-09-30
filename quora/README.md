# Quora Question Pairs, duplicated questions prediction 



My signal xgboost model solution for the [Quora Question Pairs kaggle competition] (https://www.kaggle.com/c/quora-question-pair). reach around logos 0.14370 on Public LB (~167th) / 0.14718 on Private LB (~170th). 

# features

- jaccard_ngram
- dicedistence_ngram
- compression_dist
- compression_dist_ngram
- edit_dist_aggby_["mean", "max", "min", "median"]
- count_close_ngram
- cooccurrence_ngram
- cooc_tfidf_ngram
- bm25
- LSA_Ngram
- nmf
- svd
- TSNE_LSA_Ngram
- LSA_Ngram_Cooc
- LSA_Ngram_CosineSim
- WordNet_Similarity

Training models with different subset of features and label post-processing could boost result to top rank 1%