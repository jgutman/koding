import sys, time, os, argparse
import logging, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from numpy.random import RandomState
from gensim import models
from gensim.models import word2vec

from sklearn.utils import shuffle
from SVMtrain import SVMtrain as svm

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from evaluation import readProbability
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

def makeFeatureVec(words, model, num_features, weights = None, word_index = None):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,) , dtype=np.float32)
    nwords = 0.
    if (len(words) == 0):
        return featureVec
    
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    
    # Loop over each word in the reddit post and, if it is in the model's
    # vocabulary, add its feature vector to the total
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            if (weights == None):
                featureVec = np.add(featureVec,model[word])
            else:
                # calculate weights and multiply model[word] by weight before adding
                idx = word_index.get(word)
                # get weight from weights, using index from word_index dict
                sparse_weight = weights.getcol(idx)
                if (sparse_weight.nnz != 0):
                    weight = sparse_weight.data[0]
                    featureVec = np.add(featureVec, np.multiply(model[word], weight))   

    if nwords == 0.:
        nwords = 1.
    # Divide the result by the number of words to get the average
    featureVec = np.true_divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(documents, model, num_features, weights = None, word_index = None):
    # Given a collection of documents (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    
    counter = -1
    
    # Preallocate a 2D numpy array, for speed
    docFeatureVecs = np.zeros((len(documents), num_features), dtype=np.float32)
    
    for post in documents:
    # Print a status message every 1000th review
        counter = counter + 1
        if (counter%10000 == 0):
            sys.stdout.write("Reddit post %d of %d\n" % (counter, len(documents)))
            sys.stdout.flush()
        if (weights == None):
            docFeatureVecs[counter] = makeFeatureVec(post, model, num_features)
        else:
            weightPost = weights.getrow(counter) # sparse row vector (1, size of vocabulary)
            docFeatureVecs[counter] = makeFeatureVec(post, model, num_features, 
                weights = weightPost, word_index = word_index)          
    
    return docFeatureVecs

def docWordList(text, remove_stopwords = False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    words = text.split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]    
    return words

def computeAverage(args, train, val, test, model, file_train_out, file_val_out, file_test_out):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    word_vectors = model.syn0
    vocabulary_size = int(word_vectors.shape[0])
    num_features = int(word_vectors.shape[1])
    sys.stdout.write('vocab: %d num features: %d\n' % (vocabulary_size, num_features))
    sys.stdout.flush()
    
    # Should we remove stopwords here or just implement the tf-idf weighting scheme?
    logging.info('Constructing word lists for all documents')
    train_words = []
    val_words = []
    test_words = []
    sys.stdout.write("processing stop words, building train, val, test vocabularies...\n"); sys.stdout.flush()
    for post in train['text'].astype(str):
        train_words.append(docWordList(post, remove_stopwords = args.removeStopwords))
    for post in val['text'].astype(str):
        val_words.append(docWordList(post, remove_stopwords = args.removeStopwords))
    for post in test['text'].astype(str):
        test_words.append(docWordList(post, remove_stopwords = args.removeStopwords))
    
    if args.weightedw2v:
        logging.info('Taking the weighted average')
        # Build tf-idf matrix on training documents
        logging.info('TF-IDF')
        sys.stdout.write("fitting tf-idf matrix on train...\n"); sys.stdout.flush()
        tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
            stop_words = ('english' if args.removeStopwords else None))
        tfidf_matrix =  tf.fit_transform(train['text'].astype(str))
        vocabulary = tf.vocabulary_
        sys.stdout.write("tf-idf matrix train %s\n" % str(tfidf_matrix.shape))
        logging.info('Weighted average train')
        sys.stdout.write("averaging word embeddings in training data...\n"); sys.stdout.flush()
        trainDataVecs = getAvgFeatureVecs(train_words, model, num_features,
            weights = tfidf_matrix, word_index = vocabulary)
        sys.stdout.write("writing train embeddings to %s\n" % file_train_out); sys.stdout.flush()
        trainDataVecs.dump(file_train_out)
        
        # Build tf-idf matrix on validation documents
        logging.info('TF-IDF')
        sys.stdout.write("fitting tf-idf matrix on validation...\n"); sys.stdout.flush()
        tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
            stop_words = ('english' if args.removeStopwords else None))
        tfidf_matrix =  tf.fit_transform(val['text'].astype(str))
        vocabulary = tf.vocabulary_
        sys.stdout.write("tf-idf matrix val %s\n" % str(tfidf_matrix.shape))
        logging.info('Weighted average validation')
        sys.stdout.write("averaging word embeddings in validation data...\n"); sys.stdout.flush()
        valDataVecs = getAvgFeatureVecs(val_words, model, num_features,
            weights = tfidf_matrix, word_index = vocabulary)
        sys.stdout.write("writing val embeddings to %s\n" % file_val_out); sys.stdout.flush()
        valDataVecs.dump(file_val_out)
        
        # Build tf-idf matrix on testing documents
        logging.info('TF-IDF')
        sys.stdout.write("fitting tf-idf matrix on test...\n"); sys.stdout.flush()
        tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
            stop_words = ('english' if args.removeStopwords else None))
        tfidf_matrix =  tf.fit_transform(test['text'].astype(str))
        vocabulary = tf.vocabulary_
        sys.stdout.write("tf-idf matrix test %s\n" % str(tfidf_matrix.shape))
        logging.info('Weighted average test')
        sys.stdout.write("averaging word embeddings in test data...\n"); sys.stdout.flush()
        testDataVecs = getAvgFeatureVecs(test_words, model, num_features,
            weights = tfidf_matrix, word_index = vocabulary)
        sys.stdout.write("writing test embeddings to %s\n" % file_test_out); sys.stdout.flush()
        testDataVecs.dump(file_test_out)
        
    else:
        logging.info('Weighted average train')
        sys.stdout.write("averaging word embeddings in training data...\n"); sys.stdout.flush()
        trainDataVecs = getAvgFeatureVecs(train_words, model, num_features)
        sys.stdout.write("writing train embeddings to %s\n" % file_train_out); sys.stdout.flush()
        trainDataVecs.dump(file_train_out)
        
        logging.info('Weighted average validation')
        sys.stdout.write("averaging word embeddings in validation data...\n"); sys.stdout.flush()
        valDataVecs = getAvgFeatureVecs(val_words, model, num_features)
        sys.stdout.write("writing val embeddings to %s\n" % file_val_out); sys.stdout.flush()
        valDataVecs.dump(file_val_out)
        
        logging.info('Weighted average test')
        sys.stdout.write("averaging word embeddings in testing data...\n"); sys.stdout.flush()
        testDataVecs = getAvgFeatureVecs(test_words, model, num_features)
        sys.stdout.write("writing test embeddings to %s\n" % file_test_out); sys.stdout.flush()
        testDataVecs.dump(file_test_out)
    
    logging.info('Averaging completed. Embeddings stored successfully.')
    return trainDataVecs, valDataVecs, testDataVecs

def trainw2v( context, dims, w2vpath, tokenized , cores = 4, epochs = 10):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    documents = word2vec.LineSentence(tokenized)

    # Train word2vec model with chosen parameters
    model_w2v = models.Word2Vec( documents, size=dims, window=context, min_count=10, workers=cores,
        negative=10 , sample = 0, sg = 1, iter = epochs) 

    # save embeddings to disk
    filename = format("w2v_context_%d_dim_%d.pickle" % (context, dims))
    fullpath = os.path.join(w2vpath, filename)
    sys.stdout.write("Dumping embeddings to disk at %s\n" % fullpath); sys.stdout.flush()
    model_w2v.save(fullpath)
    sys.stdout.write("All done. Vectors saved to %s\n" % fullpath); sys.stdout.flush()
    return model_w2v

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-context", dest = "context", type = int)
    parser.add_argument("-dims", dest = "dims", type = int)
    parser.add_argument("-root", dest = "root_dir")
    parser.add_argument("-data", dest = "data_path")
    parser.add_argument("-w2v", dest = "store_w2v")
    parser.add_argument("-out", dest = "store_out")
    parser.add_argument("-sentences", dest = "tokenized_text")
    parser.add_argument("-cores", dest = "cores", type = int)
    parser.add_argument("-epochs", dest = "epochs", type = int)
    parser.add_argument("-weighted", dest = "weightedw2v", action = "store_true")
    parser.add_argument("-stopwords", dest = "removeStopwords", action = "store_true")
    
    parser.set_defaults(context = 5, dims = 100, cores = 4, epochs = 10,
        weightedw2v = False, removeStopwords = False,
        root_dir = "/home/cusp/rn1041/snlp/reddit/nn_reddit",
        data_path = "d2vtune/d2v_data/",
        store_w2v = "w2vtune/embeddings/",
        store_out = "w2vtune/predictions/",
        tokenized_text = "d2vtune/tokenized.txt")
    args = parser.parse_args()
    return args

class argdict:
    def __init__(self, context=5, dims=100):
        self.context = context
        self.dims = dims
        self.cores = 4
        self.epochs = 10
        #self.root_dir = "./"
        self.root_dir = "/home/cusp/rn1041/snlp/reddit/nn_reddit"
        #self.root_dir = "/Users/jacqueline/Google Drive/gdrive/"
        self.data_path = "d2vtune/d2v_data/"
        self.store_w2v = "w2vtune/embeddings/"
        self.store_out = "w2vtune/predictions/"
        self.tokenized_text = "d2vtune/tokenized.txt"
        self.weightedw2v = True
        self.removeStopwords = True

def main(args):
    store_w2v = os.path.join(os.path.abspath(args.root_dir), args.store_w2v)
    store_out = os.path.join(os.path.abspath(args.root_dir), args.store_out)
    tokenized_path = os.path.join(os.path.abspath(args.root_dir), args.tokenized_text)
    subset_path = os.path.join(os.path.abspath(args.root_dir), args.data_path)
    
    datapath = os.path.join(subset_path, 'data.txt')
    trainpath = os.path.join(subset_path, 'train.txt')
    valpath = os.path.join(subset_path, 'val.txt')
    testpath = os.path.join(subset_path, 'test.txt')
    
    # parse pre-split, pre-tokenized data
    data = pd.read_csv( datapath, sep = '\t', header = None, 
            names = ['label', 'score', 'text'] ).dropna().reset_index(drop=True)
    train = pd.read_csv( trainpath, sep = '\t', header = None,
            names = ['label', 'score', 'text'], index_col = 0 )
    val = pd.read_csv( valpath, sep = '\t', header = None,
            names = ['label', 'score', 'text'], index_col = 0 )
    test = pd.read_csv( testpath, sep = '\t', header = None,
            names = ['label', 'score', 'text'], index_col = 0 )        

    sys.stdout.write("loaded data\n train: %s\n val: %s\n test: %s\n" % 
            (str(train.shape), str(val.shape), str(test.shape)))
    
    # build word embeddings across all documents
    model = trainw2v( args.context, args.dims, w2vpath = store_w2v, tokenized = tokenized_path, 
        cores = args.cores, epochs = args.epochs )
    
    # average word embeddings for each subset of data
    logging.info('building word vectors')
    weighted = "tf_wts" if args.weightedw2v else "unweighted"
    stopwords = "stops_removed" if args.removeStopwords else "none_removed"
    
    trainEmbedPath = os.path.join(store_w2v, format("train_context_%d_dims_%d_%s_%s.pickle" %
        (args.context, args.dims, weighted, stopwords)))
    valEmbedPath = os.path.join(store_w2v, format("val_context_%d_dims_%d_%s_%s.pickle" %
        (args.context, args.dims, weighted, stopwords)))
    testEmbedPath = os.path.join(store_w2v, format("test_context_%d_dims_%d_%s_%s.pickle" %
        (args.context, args.dims, weighted, stopwords)))
    
    trainDataVecs, valDataVecs, testDataVecs = computeAverage(args, train, val, test, model, 
        file_train_out = trainEmbedPath, file_val_out = valEmbedPath, file_test_out = testEmbedPath)
    sys.stdout.write("%d training posts, %d features\n" % (len(trainDataVecs), len(trainDataVecs[0])))
    sys.stdout.write("%d validation posts, %d features\n" % (len(valDataVecs), len(valDataVecs[0])))
    sys.stdout.write("%d test posts, %d features\n" % (len(testDataVecs), len(testDataVecs[0])))
    sys.stdout.flush()
    
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    val['y'] = le.transform(val.label)
    test['y'] = le.transform(test.label)
    logging.info('Fitting the SVM')
    filename = format("w2v_decision_function_context_%d_dim_%d_%s_%s.csv" % 
        (args.context, args.dims, weighted, stopwords))
    store_out = os.path.join(store_out, filename)
    
    svm( trainVecs, list(train.y), valVecs, list(val.y), testVecs, list(test.y), 
        le_classes_ = le.classes_, outfile = store_out, cores = args.cores )
    sys.stdout.write("Prediction matrix written to %s\n" % store_out); sys.stdout.flush()
    
    # Call evaluation script
    filename = format("confusion_plots/svm_w2v_context_%d_dim_%d_%s_%s.png" % 
        (args.context, args.dims, weighted, stopwords))
    confusion_path = os.path.join(os.path.abspath(args.root_dir), filename)
    readProbability(store_out, index = False, svm = True, datapath = datapath, outpath = confusion_path)

if __name__ == '__main__':
    sys.stdout.write("start!\n"); sys.stdout.flush()
    stime = time.time()
    args = parseArgs()
    # args = argdict()
    main(args)
    sys.stdout.write("done!\n"); sys.stdout.flush()
    etime = time.time()
    lapse = etime - stime
    sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()
