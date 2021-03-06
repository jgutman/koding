import operator, sys, random, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
import statsmodels.formula.api as smf
from TrainTest import Split



def traintest(path):
    train, test = Split(path)
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    test['y'] = le.transform(test.label)
    # sample / DELETE
    sample = train.iloc[random.sample(np.arange(0, train.shape[0]), 10000)]
    return train, test, sample, le.classes_
    print 'traintest'


def DimReduce(train, test, sample):
    # convert to dense matrix
    print 'vectorizing...'
    count_vect = CountVectorizer(min_df=10, ngram_range=(1, 3))
    train_vect = count_vect.fit(sample.text.values)
    train_count = train_vect.transform(sample.text.values)
    test_count = train_vect.transform(test.text.values)
    print 'train_count dims:', train_count.shape
    # find the best number of components base on when it hits 0.8 total variance
    trigger = True
    cnt = 0
    dim = np.linspace(1, train_count.shape[1], 100)
    nbatch = train_count.shape[0] / 1000
    batch_loop = np.linspace(0, train_count.shape[0], nbatch).astype(int)
    test_loop = np.linspace(0, test_count.shape[0], nbatch).astype(int)
    print 'searching for the best pca dims...'
    while trigger:
        columns = dim[cnt].astype(int)
        pca = IncrementalPCA(n_components=columns)
        for i, v in enumerate(batch_loop):
            if i==0:
                continue
            subset = train_count[batch_loop[i-1]:batch_loop[i]].toarray()
            pca.partial_fit(subset)
        print 'trying pca at dim:', columns, 'score:', pca.explained_variance_ratio_.sum()
        if pca.explained_variance_ratio_.sum() >= 0.81:
            print 'transforming to dim', columns
            pca_train = None
            pca_test = None
            for i, v in enumerate(batch_loop):
                if i==0:
                    continue
                temptrans = pca.transform(train_count[batch_loop[i-1]:batch_loop[i]].toarray())
                temptest = pca.transform(test_count[test_loop[i-1]:test_loop[i]].toarray())
                if pca_train == None:
                    pca_train = temptrans
                    pca_test = temptest
                else:
                    pca_train = np.vstack((pca_train, temptrans))
                    pca_test = np.vstack((pca_test, temptest))
            trigger = False
        else:
            cnt += 1
            pca = None
    # move the data out
    print 'moving the data out'
    dftrain = pd.DataFrame(pca_train)
    dftest = pd.DataFrame(pca_test)
    dftrain['y'] = train.y
    dftest['y'] = test.y
    dftrain.to_csv('pca_train.csv', index=False)
    dftest.to_csv('pca_test.csv', index=False)
    return train_count, test_count, pca_train, pca_test
    print 'PCA', '\n'


def LogitModel(pca_train, train_y, pca_test, test_y, lamb, zoom, le_classes_):
    '''
    arguments: lamb = number of values in the range.
               zoom = number of lambda value zoom ins
                      plus and minus the max score the
                      previous iteration.
    '''
    lower = 1e-6
    upper = 10
    for level in xrange(zoom):
        lambda_range = np.linspace(lower, upper, lamb)
        nested_scores = []
        for i, v in enumerate(lambda_range):
            logit = LogisticRegression(C=v)
            model = logit.fit(pca_train, train_y)
            nested_scores.append(model.score(pca_test, test_y))
            print 'level:', level, 'lambda:', v, 'score:', model.score(pca_test, test_y)
        best = np.argmax(nested_scores)
        # update the lower and upper bounds
        if best == 0:
            lower = lambda_range[best]
            upper = lambda_range[best+1]
        elif best == lamb-1:
            lower = lambda_range[best-1]
            upper = lambda_range[best]
        else:
            lower = lambda_range[best-1]
            upper = lambda_range[best+1]
        print 'best:', best, 'lambda_range', lambda_range, 'scores', nested_scores
    logit = LogisticRegression(C=lambda_range[best])
    model = logit.fit(pca_train, train_y)
    df = pd.DataFrame(model.predict_proba(pca_test), columns=[v+"_"+str(i) for i,v in enumerate(le_classes_)])
    df['y'] = test_y
    df.to_csv('predict_proba_pca_logit.csv')
    print 'FINAL SCORE', model.score(pca_test, test_y), '\n'


if __name__ == '__main__':
    print 'start'
    stime = time.time()
    script, path = sys.argv
    train, test, sample, cats = traintest(path = path)
    train_count, test_count, pca_train, pca_test = DimReduce(train, test, sample)
    LogitModel(pca_train, train.y, pca_test, test.y, 20, 5, cats)
    print 'done!'
    etime = time.time()
    ttime = etime - stime
    print ttime % 60

