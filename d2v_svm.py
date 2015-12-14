import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from gensim import models
import sys




def LoadData(d2v_train_path, d2v_test_path, train2, test2):
    sys.stdout.write('loading data...')
    sys.stdout.flush()
    d2v_train = models.Doc2Vec.load(d2v_train_path)
    vectors = d2v_train.docvecs
    train_X = np.zeros((len(vectors), len(vectors[0])))
    for i in xrange(len(vectors)):
        train_X[i] = vectors[i]
    sys.stdout.write(str(train_X[999]) + '\n')
    sys.stdout.flush()
    test_X = np.load(d2v_test_path)
    df_train = pd.read_csv(train2, sep='\t', header=None, names=['label', 'score', 'text'])
    df_test = pd.read_csv(test2, sep='\t', header=None, names=['label', 'score', 'text'])
    sys.stdout.write('encoding...\n')
    sys.stdout.flush()
    le = LabelEncoder()
    le.fit(df_train.label)
    train_Y = le.transform(df_train.label)
    test_Y = le.transform(df_test.label)
    print le.classes_
    return train_X, test_X, train_Y, test_Y, le.classes_




def SVMModelDense(train_X, train_Y, pca_test, test_y, lamb, zoom, le_classes_):
    '''
    arguments: lamb = number of values in the range.
               zoom = number of lambda value zoom ins
                      plus and minus the max score the
                      previous iteration.
    '''
    val_X, pca_test, val_Y, test_y = train_test_split(train_X, train_Y, test_size=0.2, random_state=83, stratify=train_Y.tolist())
    sys.stdout.write('train_count dims: ' +  str(train_X.shape) + '\n')
    sys.stdout.write('validation_count dims: ' +  str(val_X.shape) + '\n')
    sys.stdout.write('test_count dims: ' +  str(pca_test.shape) + '\n')
    sys.stdout.write('validation_bins dims: ' +  str(np.bincount(val_Y)) + '\n')
    sys.stdout.write('test_bins dims: ' +  str(np.bincount(test_y)) + '\n')
    sys.stdout.flush() 
    train_X = normalize(train_X, axis=0)
    val_X = normalize(val_X, axis=0)
    pca_test = normalize(pca_test, axis=0)
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
                                class_weight=weights)
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
                        class_weight=weights)
    model = clf.fit(train_X, train_Y)
    df = pd.DataFrame(model.decision_function(pca_test), 
                      columns=[v+"_"+str(i) for i,v in enumerate(le_classes_)])
    df['y'] = test_y
    df['predict'] = model.predict(pca_test)
    df.to_csv('d2v_margins.csv', index=False)
    sys.stdout.write('FINAL SCORE ' + str(model.score(pca_test, test_y)) + '\n')
    sys.stdout.flush()



if __name__ == '__main__':
    script, d2v_train_path, d2v_test_path, train2, test2 = sys.argv
    sys.stdout.write('START\n')
    sys.stdout.flush()
    train_X, test_X, train_Y, test_Y, cat = LoadData(d2v_train_path, d2v_test_path, train2, test2)
    SVMModelDense(train_X, train_Y, test_X, test_Y, 10, 10, cat)
    sys.stdout.write('FINISH\n')
    sys.stdout.flush()






