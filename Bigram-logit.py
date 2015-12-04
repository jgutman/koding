'''
Bigram model with logistic Logistic
'''

import pandas as pd
import numpy as np
from TrainTest import Split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import random, sys, time


def Bigram(path):
    train, test = Split(path)
    print test.shape, train.shape

    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    test['y'] = le.transform(test.label)

    # find the best lambda term 'C' three bags
    lamb = [0.7, 0.8, 0.9, 1.0]
    errors = []
    for j in xrange(3):
        innerList = []
        idx = random.sample(np.arange(0, train.shape[0]), 100000)
        sample = train.iloc[idx]
        count_vect = CountVectorizer(min_df=5, ngram_range=(2,2))
        X_train_counts = count_vect.fit_transform(sample.text.tolist())
        for i in lamb:
            logit = LogisticRegression(C=i)
            model = logit.fit(X_train_counts, sample.y.values)
            test_matrix = count_vect.transform(test.text.values)
            innerList.append(model.score(test_matrix, test.y.values))
        errors.append(innerList)
    best = np.array(errors).mean(axis=0).argmax()

    # dense matrix
    train_vect = CountVectorizer(min_df=5, ngram_range=(2,2))
    train_counts = train_vect.fit_transform(train.text.tolist())
    print 'count_vect size:', train_counts.shape

    # train and test
    logit = LogisticRegression(C=lamb[best])
    tmodel = logit.fit(train_counts, train.y.values)
    test_m = train_vect.transform(test.text.values)
    print 'Test Score:', tmodel.score(test_m, test.y.values)
    print 'In Sample Score:', tmodel.score(train_counts, train.y.values)

    # print out the class probabilities
    df = pd.DataFrame(tmodel.predict_proba(test_m), columns=[v+"_"+str(i) for i,v in enumerate(le.classes_)])
    df['y'] = test.y
    df.to_csv('predict_proba_bigram.csv')

if __name__ == '__main__':
    print 'start'
    stime = time.time()
    script, path = sys.argv
    Bigram(path)
    print 'done!'
    etime = time.time()
    ttime = etime - stime
    print ttime % 60

