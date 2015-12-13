import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
import sys, operator, random
from numpy.random import RandomState

def ParseData(path):
    '''
    arguments: path = path to data.txt
    returns: an array of data
    '''
    tempD = []
    skip_count = 0
    rows_dropped = 0
    with open(path) as f:
        try:
            for line in f:
                row = line.split('\t')
                label = row[0]
                score = row[1]
                text = ' '.join(row[2:]).replace("\t", " ").strip()
                if label != 'nan':
                    tempD.append([label, score, text])
                else:
                    rows_dropped+=1
        except:
            skip_count +=1
    sys.stdout.write('\rerrors catched: %d, rows dropped: %d, return size: %d\n' % (skip_count, rows_dropped, len(tempD)))
    return tempD

def Split(path, data = None, parse = True, random_seed = 83):
    '''
    arguments: path= path to data.txt
    returns: train and test split in panda DataFrame
    '''
    
    if parse:
    	D = ParseData(path)
    	data = pd.DataFrame(D, columns = ['label', 'score', 'text'])
    
    groups = data.label.unique()
    group_size = data.label.value_counts()
    
    strat_size = 20000
    random_state = RandomState(seed = random_seed)
    
    # training and test
    train = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
    test = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
    for i in groups:
    	subtrain, subtest = train_test_split(data[data.label == i], 
    		test_size = strat_size, random_state = random_state)
    	train = train.append(subtrain)
    	test = test.append(subtest)
    
    order_train = random_state.permutation(train.index)
    order_test = random_state.permutation(test.index)
    
    train = train.loc[order_train]
    test = test.loc[order_test]
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    return train, test  
 

