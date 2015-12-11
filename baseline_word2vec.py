from gensim import models
import argparse, os
import nltk

from TrainTest import Split
from TrainTest import ParseData
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
import random, sys, time

def write_training(datapath, trainpath, testpath):
	#train, test = Split(datapath)
	data = pd.read_csv(datapath, sep = '\t', header=None, names = ['label', 'score', 'text'])
	train, test = Split(datapath, data = data, parse = False)
	train.to_csv(trainpath, sep = '\t', header = False, index = False)
	test.to_csv(testpath, sep = '\t', header = False, index = False)
	return train, test

def checkfiles():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	trainpath = os.path.join(google_drive, 'data/train.txt')
	testpath = os.path.join(google_drive, 'data/test.txt')
	ParseData(trainpath)
	ParseData(testpath)
	
def tokenize(textdf):
    tokens = nltk.tokenize_sents(textdf['text'])
    stems = []
    for post in tokens:
    	for item in post:
        	stems.append(PorterStemmer().stem(item))
    return stems

# Code below based on Kaggle competition: Bag of Words Meets Bags of Popcorn
# Credited to Angela Chapman https://github.com/wendykan/DeepLearningMovies/

def makeFeatureVec(words, model, num_features, weights = None, word_index = None):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,) , dtype=np.float32)
    nwords = 0.
    
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
	
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
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
			print "Reddit post %d of %d" % (counter, len(documents))
		if (weights == None):
			docFeatureVecs[counter] = makeFeatureVec(post, model, snum_features)
		else:
			weightPost = weights.getrow(counter) # sparse row vector (1, size of vocabulary)
			docFeatureVecs[counter] = makeFeatureVec(post, model, num_features, 
				weights = weightPost, word_index = word_index)  		
     	
	return docFeatureVecs

def baselineWord2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
	# Extend Richard's baseline() function in baseline.py to use trainDataVecs / testDataVecs
	# instead of count vectorizer
	
	# encode labels
	le = LabelEncoder()
	le.fit(train.label)
	train['y'] = le.transform(train.label)
	test['y'] = le.transform(test.label)
	
	# train model
	logit = LogisticRegression()
	model = logit.fit(trainDataVecs, train.y.values)
	
	print 'Test sample score: %s' % str(model.score(testDataVecs, test.y.values))
	print 'In sample scores: %s' % str(model.score(trainDataVecs, train.y.values))

	outfile = os.path.join(outputPath, 'word2vec_logit_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	print 'CLASSES', le.classes_ 

def svmWord2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
	# Extend Richard's baseline() function in baseline.py to use trainDataVecs / testDataVecs
	# instead of count vectorizer
	
	# encode labels
	le = LabelEncoder()
	le.fit(train.label)
	train['y'] = le.transform(train.label)
	test['y'] = le.transform(test.label)
	
	# train model
	C = 1.0  # SVM regularization parameter
	model = svm.SVC(kernel='linear', C=C).fit(trainDataVecs, train.y.values)
	
	print 'Test sample score: %s' % str(model.score(testDataVecs, test.y.values))
	print 'In sample scores: %s' % str(model.score(trainDataVecs, train.y.values))

	outfile = os.path.join(outputPath, 'word2vec_svm_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	print 'CLASSES', le.classes_ 

def docWordList(text, remove_stopwords = False, to_lower = False):
	# Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    if to_lower:
    	words = text.lower().split()
    else:
    	words = text.split()
    if remove_stopwords:
    	stops = set(stopwords.words("english"))
    	words = [w for w in words if not w in stops]  	
    return words
    
def main():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	
	parser = argparse.ArgumentParser(description = 'Get word2vec model path')
	parser.add_argument('-w2v', dest = 'w2vpath', help = 'location of pre-built word2vec model')
	parser.add_argument('-train', dest = 'trainpath', help = 'location of pre-split training data')
	parser.add_argument('-test', dest = 'testpath', help = 'location of pre-split test data')
	parser.add_argument('-data', dest = 'datapath', help = 'location of unsplit data file')
	parser.add_argument('-size', dest = 'numSamples', 
		help = 'how many samples to use in the training', type = int)
	parser.add_argument('-split', dest = 'splitdata', 
		help = 'split data into train and test?', action = 'store_true')
	parser.add_argument('-weighted', dest = 'weightedw2v', 
		help = 'use tf-idf weighting for words', action = 'store_true')
	parser.add_argument('-stopwords', dest = 'removeStopWords', 
		help = 'remove English stop words', action = 'store_true')
	
	parser.set_defaults(w2vpath = os.path.join(google_drive, 'w2v_output1/w2v_train_only.txt'), 
		trainpath = os.path.join(google_drive, 'data/train2.txt'), 
		testpath = os.path.join(google_drive, 'data/test2.txt'),
		datapath = os.path.join(google_drive, 'data3.txt'), 
		splitdata = False, weightedw2v = False, removeStopWords = False, size = 0)
	args = parser.parse_args()
	datapath = os.path.abspath(args.datapath)
	trainpath = os.path.abspath(args.trainpath)
	testpath = os.path.abspath(args.trainpath)
	w2vpath = os.path.abspath(args.w2vpath)
	
	print "loading word2vec..."
	model = models.Word2Vec.load(w2vpath)
	print "loading train and test data..."
	if args.splitdata:
		train, test = write_training(datapath, trainpath, testpath)
	else:
		train = pd.read_csv(trainpath, 
			sep = '\t', header=None, names = ['label', 'score', 'text'])
		test = pd.read_csv(testpath, 
			sep = '\t', header=None, names = ['label', 'score', 'text'])
	
	word_vectors = model.syn0
	vocabulary_size = int(word_vectors.shape[0])
	num_features = int(word_vectors.shape[1])
	print vocabulary_size, num_features

	# Should we remove stopwords here or just implement the tf-idf weighting scheme?
	train_words = []
	test_words = []
	print "processing stop words, building train and test vocabularies..."
	for post in train['text'].astype(str):
		train_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	for post in test['text'].astype(str):
		test_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	
	if args.weightedw2v:
		# Build tf-idf matrix on training documents
		print "fitting tf-idf matrix..."
		tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
			stop_words = ('english' if args.removeStopWords else None))
		tfidf_matrix_train =  tf.fit_transform(train['text'].astype(str))
		vocabulary = tf.vocabulary_
		print tfidf_matrix_train.shape
		print "averaging word embeddings in training data..."
		trainDataVecs = getAvgFeatureVecs(train_words, model, num_features,
			weights = tfidf_matrix_train, word_index = vocabulary)
		
		# Apply tf-idf matrix from training to test documents to get weights
		print "averaging word embeddings in test data..."
		testDataVecs = getAvgFeatureVecs(test_words, model, num_features,
			weights = tfidf_matrix_train, word_index = vocabulary)
		
	else:
		print "averaging word embeddings in training data..." 
		trainDataVecs = getAvgFeatureVecs(train_words, model, num_features)
		# write the word embeddings to file so we can read in quickly
		file_train_out = os.path.join(os.path.dirname(trainpath), 'train_word_embeddings.txt')
		np.savetxt(file_train_out, trainDataVecs, delimiter='\t')
		
		print "averaging word embeddings in test data..."
		testDataVecs = getAvgFeatureVecs(test_words, model, num_features)
		file_test_out = os.path.join(os.path.dirname(testpath), 'train_word_embeddings.txt')
		np.savetxt(file_test_out, testDataVecs, delimiter='\t')
	
	print "fitting baseline model on averaged word embeddings..."
	outputDirectory = os.path.dirname(w2vpath)
	baselineWord2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory)
	svmWord2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory)

if __name__ == '__main__':
	print 'start'
	stime = time.time()
	main()
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime / 60