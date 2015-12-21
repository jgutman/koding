import os, sys, random, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from SVMtrain import SVMtrain as svm
from evaluation import readProbability

def traintest(path):
    trainpath = os.path.join(path, 'train.txt')
    valpath = os.path.join(path, 'val.txt')
    testpath = os.path.join(path, 'test.txt')
    train = pd.read_csv(trainpath, sep='\t', header=None, names = ['label', 'score', 'text']).dropna()
    val = pd.read_csv(valpath, sep='\t', header=None, names = ['label', 'score', 'text']).dropna()
    test = pd.read_csv(testpath, sep='\t', header=None, names = ['label', 'score', 'text']).dropna()
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    val['y'] = le.transform(val.label)
    test['y'] = le.transform(test.label)
    return train, val, test, le.classes_

def SparseMatrix(train, val, test, ngram):
    # convert to dense matrix
    sys.stdout.write('\nvectorizing...\n')
    sys.stdout.flush()
    count_vect = CountVectorizer(min_df=10, ngram_range=(1, ngram),
        stop_words = 'english', lowercase = True)
    train_count = count_vect.fit_transform(train.text.values)
    val_count = count_vect.transform(val.text.values)
    test_count = count_vect.transform(test.text.values)
    sys.stdout.write('train_count dims: %s\n' %  str(train_count.shape))
    sys.stdout.write('val_count dims: %s\n' %  str(val_count.shape))
    sys.stdout.write('test_count dims: %s\n' %  str(test_count.shape))
    sys.stdout.flush()    
    return train_count, val_count, test_count

if __name__ == '__main__':
    sys.stdout.write('START\n')
    sys.stdout.flush()
    script, path, ngram, = sys.argv
    path = os.path.abspath(path)
    train, val, test, cat = traintest(path)
    train_count, val_count, test_count, = SparseMatrix(train, val, test, int(ngram))
    outfile = os.path.join(os.path.dirname(path), 
        format('lr_proba_ngram_%d.csv' % int(ngram)))
    confusion_path = os.path.join(os.path.dirname(path), 
        format('lr_confusion_ngram_%d.png' % int(ngram)))
    sys.stdout.write("%s\n%s\n" % (outfile, confusion_path)); sys.stdout.flush()
    svm(train_count, train.y.values, val_count, val.y.values, test_count, test.y.values,
        cat, outfile)
    readProbability(outfile, index = False, svm = False, datapath = os.path.join(path, 'data.txt'), 
        outpath = confusion_path)
    sys.stdout.write('ngram: %d %s\n' % (int(ngram), outfile))