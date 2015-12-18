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

def trainw2v( data, context, dims, w2vpath, tokenized , weighted = True, stopwords = True,
    cores = 4, epochs = 10, seed = 150):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	
	documents = doc2vec.TaggedLineDocument(tokenized)
	doc_list = [doc for doc in documents]
	tag_list = [doc.tags[0] for doc in documents]

	# instantiate DM and DBOW models
	model_dbow = models.Doc2Vec( size=dims, window=context, dm=0, min_count=10, workers=cores,
		negative=10 , sample = 0) #, sample = 1e-5 )
	model_dm = models.Doc2Vec( size=dims, window=context, dm=1, min_count=10, workers=cores,
		negative=10 , sample = 0) #, sample = 1e-5 )
	
	# build vocab over all documents
	sys.stdout.write("Building vocabulary across all data\n"); sys.stdout.flush()
	model_dm.build_vocab(documents)
	model_dbow.reset_from(model_dm)
	sys.stdout.write("Vocabularies built. DM: %d words, DBOW: %d words\n" %
		(len(model_dm.vocab), len(model_dbow.vocab))); sys.stdout.flush()
	
	for epoch in range(epochs):
		stime = time.time()
		sys.stdout.write("Training doc2vec epoch %d of %d\n" % (epoch+1, epochs)); sys.stdout.flush()
		shuffled_documents = shuffle(doc_list, random_state = seed)
		seed += 1
		logging.info('Training DM model, epoch %d of %d' % (epoch+1, epochs))
		model_dm.train(shuffled_documents)
		logging.info('Training DBOW model, epoch %d of %d' % (epoch+1, epochs))
		model_dbow.train(shuffled_documents)
		lapse = time.time() - stime
		sys.stdout.write("Epoch %d took (%0.0f min, %0.0f sec)\n" % 
				(epoch+1, np.floor(lapse / 60.), lapse % 60.))
	
	# combine models
	model = ConcatenatedDoc2Vec([model_dm, model_dbow])
	np_model = np.vstack((model.docvecs[tag] for tag in tag_list))
	
	# save embeddings to disk
	filename = format("d2v_context_%d_dim_%d_dm_dbow.pickle" % (context, dims))
	fullpath = os.path.join(d2vpath, filename)
	sys.stdout.write("Dumping embeddings to disk at %s\n" % fullpath); sys.stdout.flush()
	if (dims > 200):
		path1 = format("%s.part1" % fullpath)
		path2 = format("%s.part2" % fullpath)
		partition = 500000
		np_model[:partition, :].dump(path1)
		np_model[partition: , :].dump(path2)
		sys.stdout.write("All done. Vectors saved to %s and %s\n" % (path1, path2)); sys.stdout.flush()
	else:
		np_model.dump(fullpath)
		sys.stdout.write("All done. Vectors saved to %s\n" % fullpath); sys.stdout.flush()
	return np_model


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
    parser.add_argument("-weighted", dest = "weighted", action = "store_true")
    parser.add_argument("-stopwords", dest = "stopwords", action = "store_true")
    
    parser.set_defaults(context = 5, dims = 100, cores = 4, epochs = 10,
        weighted = False, stopwords = False,
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
        self.weighted = True
        self.stopwords = True

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
    
    # build word embeddings
    
    model = trainw2v( data, args.context, args.dims, w2vpath = store_w2v, tokenized = tokenized_path, 
        weighted = args.weighted, stopwords = args.stopwords, cores = args.cores, epochs = args.epochs )
    
    # train SVM on averaged word embeddings
    trainVecs = model[train.index, : ]
    valVecs = model[val.index, : ]
    testVecs = model[test.index, : ]
    
    # encode labels
    le = LabelEncoder()
    le.fit(train.label)
    train['y'] = le.transform(train.label)
    val['y'] = le.transform(val.label)
    test['y'] = le.transform(test.label)
    logging.info('Fitting the SVM')
    weighted = "tf_wts" if args.weighted else "unweighted"
    stopwords = "stops_removed" if args.stopwords else "none_removed"
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
