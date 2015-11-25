'''
Some useful scripts for parsing the sentences and data
'''

import pandas as pd
import numpy as np
import sys, datetime

# Parse cleaned data into a vector
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
    sys.stdout.write('\rerrors catched: %d, rows dropped: %d' % (skip_count, rows_dropped))
    return tempD

# Create a file of just sentences for NN, break at '.'
def Sentences(strings, save=False):
    '''
    arguement: strings, array of strings
    returns: nested array, first position index of string array, 
    second position sentence at '.'
    '''
    now = datetime.datetime.now().strftime('%y%m%d-%H-%M-%S') + '.txt'
    output = open(now, 'w')
    sentence_count = 0
    sen = []
    for idx, text in enumerate(strings):
        sentences = text.split('.')
        for line in sentences:
            if str(line).strip():
            	sen.append([idx, str(line).strip()])
                sentence_count+=1
                if save:
                	output.write(str(line).strip()+"\n")
    output.close()
    if save:
    	sys.stdout.write('\rfile saved to ' + now + ' number of sentences: %d' % sentence_count)
    else:
	    sys.stdout.write('\rnumber of sentences: %d' % sentence_count)
    return sen

