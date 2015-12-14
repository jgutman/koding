import pandas as pd
import numpy as np
train = pd.read_csv('./data/train2.txt', sep='\t', header=None, names=['label','score','text'])
train.shape
train.dropna().shape
trainvecs = np.load('./data/train_word_embeddings.pickle')
trainvecs.shape
indices = np.where(np.any(np.isnan(trainvecs), axis = 1))[0]