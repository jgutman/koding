import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import sys, operator, random

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
    sys.stdout.write('\rerrors catched: %d, rows dropped: %d, return size: %d' % (skip_count, rows_dropped, len(tempD)))
    return tempD


def Split(path):
    '''
    arguments: path= path to data.txt
    returns: train and test split in panda DataFrame
    '''
    D = ParseData(path)
    
    # count of reddit types
    Counter = {}
    for i in D:
        text = i[0]
        if text not in Counter:
            Counter[text] = 1
        else:
            Counter[text] += 1
            
    # nested dictionary of reddit types
    groups = {i:[] for i in Counter.keys()}
    for j in D:
        groups[j[0]].append(j)
            
    # training and test
    train = {i:[] for i in Counter.keys()}
    test = {i:[] for i in Counter.keys()}
    for i in groups.keys():
        if Counter[i] > 40000:
            subtrain, subtest = train_test_split(groups[i], test_size=20000, random_state=83)
            train[i] = subtrain
            test[i] = subtest
        else:
            # those that dont have enough samples
            pass
        
    # append
    trainlist = []
    testlist = []
    for idx, key in enumerate(train.keys()):
        if train[key]:
            train_df = pd.DataFrame(train[key], columns=['label', 'score', 'text'])
            test_df = pd.DataFrame(test[key], columns=['label', 'score', 'text'])
            trainlist.append(train_df)
            testlist.append(test_df)
    
    return pd.concat(trainlist).reset_index().drop('index', 1), pd.concat(testlist).reset_index().drop('index', 1)  
 

