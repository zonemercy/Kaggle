# -*- coding: utf-8 -*-

import os
import platform
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS




RAW_PATH =  os.path.expanduser("~") + "/data/quora/"
FEAT_PATH =  os.path.expanduser("~") + "/data/quora/features/"
SUB_PATH =  os.path.expanduser("~") + "/data/quora/submission/"



# ---------------------- Overall -----------------------
TASK = "sample"
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# size
TRAIN_SIZE = 404290
TEST_SIZE = 2345796

if TASK == "sample":
	TRAIN_SIZE = SAMPLE_SIZE
	TEST_SIZE = SAMPLE_SIZE

MISSING_VALUE_NUMERIC = -1
STR_MATCH_THRESHOLD = 0.6
VALID_SIZE_MAX = 60000 # 0.7 * TRAIN_SIZE

# bm25
BM25_K1 = 1.6
BM25_B = 0.75

RANDOM_SEED = 524

# svd
SVD_DIM = 10
SVD_N_ITER = 5

#tfidf
MIN_DF = 3
MAX_DF = 0.7

oof_random = 1988
ab_dup_test =[6750,  23693,  30851,  61404,  78271, 103525, 121182, 143641,
            154513, 158473, 172120, 174071, 182820, 190035, 192380, 205866,
            211669, 220517, 236250, 240964, 251464, 252019, 254962, 272794,
            276854, 285520, 308063, 310728, 316633, 347129, 355138, 365306,
            381782, 395473, 398714, 399243]