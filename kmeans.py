import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
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
    return train_X, test_X, train_Y, test_Y




def Cluster(train_X, test_X, train_Y, test_Y):
    sys.stdout.write('clustering...\n')
    sys.stdout.flush()
    clf = MiniBatchKMeans(n_clusters=5)
    model = clf.fit(train_X)
    predict = model.predict(test_X)
    return predict




def ListScores(df_predict_values, df_y_values):
    sys.stdout.write('scoring...\n')
    sys.stdout.flush()
    count = 0.0
    labels = np.unique(df_predict_values)
    print labels
    tempDict = {i:np.zeros(len(labels)) for i in labels}
    for i, v in enumerate(df_predict_values):
        if v != df_y_values[i]:
            count+=1.0
        tempDict[v][df_y_values[i]]+=1
    print count/df.shape[0]
    return tempDict




if __name__ == '__main__':
    script, d2v_train_path, d2v_test_path, train2, test2 = sys.argv
    sys.stdout.write('START\n')
    sys.stdout.flush()
    train_X, test_X, train_Y, test_Y = LoadData(d2v_train_path, d2v_test_path, train2, test2)
    predict = Cluster(train_X, test_X, train_Y, test_Y)
    tempDict = ListScores(predict, test_Y)
    df = pd.DataFrame(tempDict)
    df.to_csv('kmean.csv', index=False)
    print tempDict






