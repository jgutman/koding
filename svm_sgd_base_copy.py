import operator, sys, random, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.kernel_approximation import RBFSampler
from TrainTest import Split


def traintest(path):
    data = pd.read_csv(path, sep='\t', header=None, names = ['label', 'score', 'text']).dropna()
    train, test = Split(path, data=data, parse=False, testsize=40000)
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    test['y'] = le.transform(test.label)
    sample = train.iloc[random.sample(np.arange(0, train.shape[0]), 10000)]
    return train, test, sample, le.classes_
    sys.stdout.write( 'traintest\n')
    sys.stdout.flush()


def SparseMatrix(train, test, sample, ngram):
    # convert to dense matrix
    sys.stdout.write('\nvectorizing...\n')
    sys.stdout.flush()
    count_vect = CountVectorizer(min_df=10, ngram_range=(1, ngram))
    train_vect = count_vect.fit(train.text.values)
    train_count = train_vect.transform(train.text.values)
    test_count = train_vect.transform(test.text.values)
    sample_count = train_vect.transform(sample.text.values)
    sys.stdout.write('train_count dims: ' +  str(train_count.shape) + '\n')
    sys.stdout.flush()    
    return train_count, test_count, sample_count


def RBFtransform(train_count, val_count, test_count, comp):
    clf = RBFSampler(gamma=2, n_components=comp, random_state=83)
    train_RBF = clf.fit_transform(train_count)
    val_RBF = clf.transform(val_count)
    test_RBF = clf.transform(test_count)
    return train_RBF, val_RBF, test_RBF


def SVMModelDense(train_X, train_Y, pca_test, test_y, lamb, zoom, le_classes_, ngram, comp, kernel=False):
    '''
    arguments: lamb = number of values in the range.
               zoom = number of lambda value zoom ins
                      plus and minus the max score the
                      previous iteration.
    '''
    val_X, pca_test, val_Y, test_y = train_test_split(pca_test, test_y, test_size=0.5, random_state=83, stratify=test_y.tolist())
    sys.stdout.write('train_count dims: ' +  str(train_X.shape) + '\n')
    sys.stdout.write('validation_count dims: ' +  str(val_X.shape) + '\n')
    sys.stdout.write('test_count dims: ' +  str(pca_test.shape) + '\n')
    sys.stdout.write('validation_bins dims: ' +  str(np.bincount(val_Y)) + '\n')
    sys.stdout.write('test_bins dims: ' +  str(np.bincount(test_y)) + '\n')
    sys.stdout.flush() 
    if kernel:
        train_X, val_X, pca_test = RBFtransform(train_X, val_X, pca_test, comp)
        print 'tx', train_X.shape, 'vx', val_X.shape
        print 'kernel true:', comp
    lower = 1e-6
    upper = 10
    # weights
    Counter = {}
    for i in train_Y:
        if i in Counter:
            Counter[i] += 1.0
        else:
            Counter[i] = 1.0
    n_sample = len(train_Y)
    n_classes = len(np.unique(train_Y))
    topCount = max(Counter.values())
    weights = {i: n_sample/(n_classes * Counter[i]) for i, v in enumerate(le_classes_)}
    print weights
    for level in xrange(zoom):
        lambda_range = np.linspace(lower, upper, lamb)
        nested_scores = []
        for i, v in enumerate(lambda_range):
            clf = SGDClassifier(alpha=v, loss='hinge', penalty='l2', 
                                l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True,  
                                learning_rate='optimal', class_weight=weights)
            model = clf.fit(train_X, train_Y)
            nested_scores.append(model.score(val_X, val_Y))
            sys.stdout.write('level: '+str(level)+' lambda: '+str(v)+' score: '+str(model.score(val_X, val_Y))+'\n')
            sys.stdout.flush()
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
        sys.stdout.write('best: ' + str(best) + ' scores ' + str(nested_scores[best]) + '\n')
        sys.stdout.flush()
    clf = SGDClassifier(alpha=lambda_range[best], loss='hinge', penalty='l2', 
                        l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True,  
                        learning_rate='optimal', class_weight=weights)
    model = clf.fit(train_X, train_Y)
    df = pd.DataFrame(model.decision_function(pca_test), 
                      columns=[v+"_"+str(i) for i,v in enumerate(le_classes_)])
    df['y'] = test_y
    df['predict'] = model.predict(pca_test)
    df.to_csv('decision_function_svm_dense_matrix_ngram-'+str(ngram)+'.csv', index=False)
    sys.stdout.write('FINAL SCORE ' + str(model.score(pca_test, test_y)) + '\n')
    sys.stdout.flush()




if __name__ == '__main__':
    sys.stdout.write('START')
    sys.stdout.flush()
    script, path, ngram, comp = sys.argv
    train, test, sample, cat = traintest(path)
    train_count, test_count, sample_count = SparseMatrix(train, test, sample, int(ngram))
    test_count.shape, sample_count.shape
    SVMModelDense(train_count, train.y.values, test_count, test.y.values, 10, 10, cat, int(ngram), int(comp), kernel=True)
    print path, 'ngram: ', ngram

