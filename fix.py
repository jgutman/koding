datapath = 'data/data3.txt'
import pandas as pd
data = pd.read_csv(datapath, sep = '\t', header = None, names = ['label', 'score', 'text'])
data.shape
data = pd.read_csv(datapath, sep = '\t', header = None, names = ['label', 'score', 'text']).dropna()
data.shape
len(data.index)
data.index