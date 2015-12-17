import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV, ParameterSampler

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        sys.stdout.write("Model with rank %d" % (i + 1))
        sys.stdout.write("Mean validation score: {0:.3f}\n" % score.mean_validation_score )
        sys.stdout.write("Parameters: %s/n" % score.parameters)

def SVMtrain(train_vecs, train_labels, val_vecs, val_labels, test_vecs, test_labels, 
    le_classes_, outfile, cores = 4, lamb = 20, zoom = 5):
    '''
    arguments: lamb = number of values in the range.
               zoom = number of lambda value zoom ins
                      plus and minus the max score the
                      previous iteration.
    '''
    sys.stdout.write('train dims: %s\n' % str(train_vecs.shape))
    sys.stdout.write('validation dims: %s\n' % str(val_vecs.shape))
    sys.stdout.write('test dims: %s\n' % str(test_vecs.shape))
    sys.stdout.write('train distribution: %s\n' % str(np.bincount(train_labels)))
    sys.stdout.write('validation distribution: %s\n' % str(np.bincount(val_labels)))
    sys.stdout.write('test distribution: %s\n' % str(np.bincount(test_labels)))
    sys.stdout.flush() 
    
    #train_vecs = normalize(train_vecs, axis=0)
    #val_vecs = normalize(val_vecs, axis=0)
    #test_vecs = normalize(test_vecs, axis=0)
    
    param_grid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 
                'penalty' = ['l2', 'l1', 'elasticnet'], 
                'alpha' = np.logspace(-4, 2, 100) 
                #'alpha' = stats.expon(scale = .1, loc = 0)
                }
    clf = SGDClassifier( n_iter=5, n_jobs=cores, verbose = True
                shuffle=True, warm_start = True, class_weight=None)
    cv = RandomizedSearchCV( clf, param_grid, n_iter = 100 )
    cv.fit(train_vecs, train_labels)
    report(cv.grid_scores_)
    
    df = pd.DataFrame(model.decision_function(test_vecs), 
            columns=[v+"_"+str(i) for i,v in enumerate(le_classes_)])
    df['y'] = test_labels
    df['predict'] = model.predict(test_vecs)
    df.to_csv(outfile, index=False)
    sys.stdout.write('FINAL SCORE: %0.4f\n' % model.score(test_vecs, test_labels))
    sys.stdout.flush()