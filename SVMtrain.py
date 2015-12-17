import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

def SVMtrain(train_vecs, train_labels, val_vecs, val_labels, test_vecs, test_labels, 
    le_classes_, outfile, lamb = 20, zoom = 5):
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
    
    train_vecs = normalize(train_vecs, axis=0)
    val_vecs = normalize(val_vecs, axis=0)
    test_vecs = normalize(test_vecs, axis=0)
    lower = 1e-4
    upper = 10
    
    for level in xrange(zoom): 
        # lambda_range = np.linspace(lower, upper, lamb)
        lambda_range = np.logspace(np.log10(lower), np.log10(upper), lamb) 
        nested_scores = [] 
        
        for i, v in enumerate(lambda_range): 
            clf = SGDClassifier(alpha=v, loss='hinge', penalty='l2', 
                l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True, warm_start = True, class_weight=None)
            model = clf.fit(train_vecs, train_labels)
            score = model.score(val_vecs, val_labels)
            nested_scores.append(score)
            sys.stdout.write("level: %d\tlambda: %0.6f\tscore: %0.4f\n" % (level, v, score))
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
        sys.stdout.write('best lambda at zoom %d: %0.6f\tscore: %0.4f\n' % 
            (level+1, lambda_range[best], nested_scores[best]))
        sys.stdout.flush()
    clf = SGDClassifier(alpha=lambda_range[best], loss='hinge', penalty='l2',
        l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True, warm_start = True, class_weight=None)
    model = clf.fit(train_vecs, train_labels)
    
    df = pd.DataFrame(model.decision_function(test_vecs), 
            columns=[v+"_"+str(i) for i,v in enumerate(le_classes_)])
    df['y'] = test_labels
    df['predict'] = model.predict(test_vecs)
    df.to_csv(outfile, index=False)
    sys.stdout.write('FINAL SCORE: %0.4f\n' % model.score(test_vecs, test_labels))
    sys.stdout.flush()