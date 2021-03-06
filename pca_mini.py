import operator, sys, random, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.cross_validation import train_test_split
import statsmodels.formula.api as smf
from sklearn.decomposition import MiniBatchSparsePCA
from TrainTest import Split



def traintest(path):
    data = pd.read_csv(path, sep='\t', header=None, names = ['label', 'score', 'text']).dropna()
    train, test = Split(path, data=data, parse=False, testsize=20000)
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    test['y'] = le.transform(test.label)
    # sample / DELETE
    sample = train.iloc[random.sample(np.arange(0, train.shape[0]), 10000)]
    return train, test, sample, le.classes_
    sys.stdout.write( 'traintest\n')
    sys.stdout.flush()


def DimReduce(train, test, sample):
    # convert to dense matrix
    sys.stdout.write('vectorizing...\n')
    sys.stdout.flush()
    count_vect = CountVectorizer(min_df=50, ngram_range=(1, 1))
    train_vect = count_vect.fit(train.text.values)
    train_count = train_vect.transform(train.text.values)
    test_count = train_vect.transform(test.text.values)
    sys.stdout.write('train_count dims: ' +  str(train_count.shape) + '\n')
    sys.stdout.flush()
    # find the best number of components base on when it hits 0.8 total variance
    trigger = True
    cnt = 0
    dim = np.linspace(1, train_count.shape[1], 10)
    nbatch = train_count.shape[0] / 1000
    batch_loop = np.linspace(0, train_count.shape[0], nbatch).astype(int)
    test_loop = np.linspace(0, test_count.shape[0], nbatch).astype(int)
    sys.stdout.write('searching for the best pca dims. len(batch_loop): '+str(len(batch_loop))+ '\n' )
    sys.stdout.flush()
    clf = MiniBatchSparsePCA(n_components=1000, alpha=1, ridge_alpha, n_iter=100, batch_size=3, n_jobs=5, method='cd', shuffle=True, random_state=83)
    model = clf.fit(train_count.toarray())
    pca_train = model.transform(train_count.toarray())
    pca_test = model.transform(test_count.toarray())
    # move the data out
    sys.stdout.write('moving the data out\n')
    sys.stdout.flush()
    dftrain = pd.DataFrame(pca_train, dtype=np.float32)
    dftest = pd.DataFrame(pca_test, dtype=np.float32)
    dftrain['y'] = train.y
    dftest['y'] = test.y
    dftrain.to_csv('pca_float32_train.csv', index=False)
    dftest.to_csv('pca_float32_test.csv', index=False)
    return train_count, test_count, pca_train, pca_test
    sys.stdout.write('PCA\n')
    sys.stdout.flush()


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
    sys.stdout.write('start\n')
    sys.stdout.flush()
    stime = time.time()
    script, path = sys.argv
    train, test, sample, cats = traintest(path = path)
    sys.stdout.write('train test Split is done\n')
    train_count, test_count, pca_train, pca_test = DimReduce(train, test, sample)
    sys.stdout.write('done!\n')
    sys.stdout.flush()
    etime = time.time()
    ttime = etime - stime
    print ttime % 60

