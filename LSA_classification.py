from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging, os, sys, time
import numpy as np
import pandas as pd
from SVMtrain import SVMtrain as svm
from evaluation import readProbability

def tf_idf_transform(train, val, test):
    vectorizer = TfidfVectorizer(min_df=10, sublinear_tf=True, 
        ngram_range = (1,3), stop_words='english', lowercase = True)
    X_train = vectorizer.fit_transform(train.text.values)
    X_val = vectorizer.transform(val.text.values)
    X_test = vectorizer.transform(test.text.values)
    return X_train, X_val, X_test

def import_data(path):
    logging.info("importing data")
    trainpath = os.path.join(path, 'train.txt')
    valpath = os.path.join(path, 'val.txt')
    testpath = os.path.join(path, 'test.txt')
    
    train = pd.read_csv(trainpath, sep='\t', header = None, names = ['label', 'score', 'text']).dropna()
    val = pd.read_csv(valpath, sep='\t', header = None, names = ['label', 'score', 'text']).dropna()
    test = pd.read_csv(testpath, sep='\t', header = None, names = ['label', 'score', 'text']).dropna()
    
    sys.stdout.write("Train: %s\nVal: %s\nTest: %s\n" % 
        (str(train.shape), str(val.shape), str(test.shape))); sys.stdout.flush()
    return train, val, test

def main(path):
    train, val, test = import_data(path)
    logging.info("fitting tf-idf")
    X_train, X_val, X_test = tf_idf_transform(train, val, test)
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    val['y'] = le.transform(val.label)
    test['y'] = le.transform(test.label)
    sys.stdout.write("Train: %s\nVal: %s\nTest: %s\n" % 
        (str(X_train.shape), str(X_val.shape), str(X_test.shape))); sys.stdout.flush()
    
    reduced_dims = 200
    logging.info(format("fitting SVD, reducing dimensions to %d\n" % reduced_dims))
    svd = TruncatedSVD(n_components=reduced_dims)
    svd.fit(X_train)
    reduced_trainX = svd.transform(X_train)
    reduced_valX = svd.transform(X_val)
    reduced_testX = svd.transform(X_test)
    sys.stdout.write("Train: %s\nVal: %s\nTest: %s\n" % 
        (str(reduced_trainX.shape), str(reduced_valX.shape), str(reduced_testX.shape))); sys.stdout.flush()
    
    reduced_trainX.dump(os.path.join(os.path.dirname(path), "LSA_tfidf_embeddings.pickle"))
    outfile = os.path.join(os.path.dirname(path), "LSA_tfidf_svm.csv")
    confusion_path = os.path.join(os.path.dirname(path), "LSA_tfidf_svm.png")
    logging.info("training SVM")
    svm(reduced_trainX, train.y.values, reduced_valX, val.y.values, reduced_testX, test.y.values, 
        le.classes_, outfile = outfile)
    logging.info("evaluating results")
    readProbability(outfile, index = False, svm = True, datapath = os.path.join(path, 'data.txt'), 
        outpath = confusion_path)

if __name__ == '__main__':
    # Display progress logs on stdout
    script, path = sys.argv
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Start!")
    stime = time.time()
    main(path)
    logging.info("Done!")
    etime = time.time()
    lapse = etime - stime
    sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush() 