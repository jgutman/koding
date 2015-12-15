from gensim import models
import argparse, os, logging
import nltk

from TrainTest2 import Split
from TrainTest2 import ParseData
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
import random, sys, time
from numpy.random import RandomState

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split

def write_training(datapath, trainpath, testpath):
	#train, test = Split(datapath)
	data = pd.read_csv(datapath, sep = '\t', header=None, names = ['label', 'score', 'text'])
	train, test = Split(datapath, data = data, parse = False)
	train.to_csv(trainpath, sep = '\t', header = False, index = False)
	test.to_csv(testpath, sep = '\t', header = False, index = False)
	logging.info('Splitting training and test')
	return train, test

def checkfiles():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	trainpath = os.path.join(google_drive, 'data/train.txt')
	testpath = os.path.join(google_drive, 'data/test.txt')
	ParseData(trainpath)
	ParseData(testpath)
	
def tokenize(textdf):
	logging.info('Tokenizing text')
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

def logitWord2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
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
	
	sys.stdout.write('Test sample score: %0.4f\n' % model.score(testDataVecs, test.y.values))
	sys.stdout.write('In sample scores: %0.4f\n' % model.score(trainDataVecs, train.y.values))
	sys.stdout.flush()

	outfile = os.path.join(outputPath, 'word2vec_logit_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	sys.stdout.write('CLASSES: %s\n' % le.classes_)
	sys.stdout.flush() 

def trainValidationSplit(data, dataY, random_seed = 100, strat_size = 20000):
	groups = data.label.unique()
	random_state = RandomState(seed = random_seed)

	# training and validation
	subtrain = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	val = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	for i in groups:
		subtrain_i, val_i = train_test_split(data[data.label == i], 
			test_size = strat_size, random_state = random_state)
		subtrain = subtrain.append(subtrain_i)
		val = val.append(val_i)
	
	return subtrain, val, dataY[subtrain.index], dataY[val.index]
	
def svmWord2Vec(train, test, trainDataVecs, testDataVecs, outputPath, lamb, zoom, 
		random_seed = 100, strat_size = 20000):
		
	# encode labels
	le = LabelEncoder()
	le.fit(train.label)
	train['y'] = le.transform(train.label)
	test['y'] = le.transform(test.label)
	
	# split training into subtraining and validation
	subtrain, val, subtrain_Y, val_Y = trainValidationSplit(train, train.y.values,
		random_seed, strat_size)
	subtrain_X = trainDataVecs[subtrain.index, :]
	val_X = trainDataVecs[val.index, :]
	
	sys.stdout.write('train_count dims: %s\n' % str(subtrain_X.shape))
	sys.stdout.write('validation_count dims: %s\n' % str(val_X.shape))
	sys.stdout.write('test_count dims: %s\n' % str(testDataVecs.shape))
	sys.stdout.write('validation_bins dims: %s\n' % str(np.bincount(val_Y)))
	sys.stdout.write('test_bins dims: %s\n' % str(np.bincount(test.y.values)))
	sys.stdout.flush() 
	
	lower = 1e-6
	upper = 10.
	
	for level in xrange(zoom):
		lambda_range = np.logspace(np.log10(lower), np.log10(upper), lamb)
		nested_scores = []
		for i, v in enumerate(lambda_range):
			clf = SGDClassifier(alpha=v, loss='hinge', penalty='l2', 
				l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True,  
				learning_rate='optimal', class_weight="balanced")
			model = clf.fit(subtrain_X, sub_train_Y)
			nested_scores.append(model.score(val_X, val_Y))
			sys.stdout.write('level: %d lambda: %0.4f score: %0.4f\n' % (level, v, model.score(val_X, val_Y)))
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
		sys.stdout.write('best: %0.4f score: %0.4f\n'  % (best, nested_scores[best]))
		sys.stdout.flush()
	clf = SGDClassifier(alpha=lambda_range[best], loss='hinge', penalty='l2', 
				l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True,  
				learning_rate='optimal', class_weight="balanced")
	model = clf.fit(subtrain_X, subtrain_Y)
	df = pd.DataFrame(model.decision_function(testDataVecs), 
		columns=[v for i,v in enumerate(le_classes_)])
	df['y'] = test.y.values
	df['predict'] = model.predict(testDataVecs)
	df.to_csv('decision_function_svm_word2vec.csv', sep='\t', index=False)
	sys.stdout.write('FINAL SCORE %0.4f\n' % model.score(testDataVecs, test.y.values))
	sys.stdout.flush()

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

def computeAverage(args, train, test, w2vpath, file_train_out, file_test_out):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sys.stdout.write("loading word2vec...\n"); sys.stdout.flush()
	model = models.Word2Vec.load(w2vpath)
	sys.stdout.write("loading train and test data...\n"); sys.stdout.flush()
	
	word_vectors = model.syn0
	vocabulary_size = int(word_vectors.shape[0])
	num_features = int(word_vectors.shape[1])
	sys.stdout.write('vocab: %d num features: %d\n' % (vocabulary_size, num_features))
	sys.stdout.flush()

	# Should we remove stopwords here or just implement the tf-idf weighting scheme?
	train_words = []
	test_words = []
	sys.stdout.write("processing stop words, building train and test vocabularies...\n"); sys.stdout.flush()
	for post in train['text'].astype(str):
		train_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	for post in test['text'].astype(str):
		test_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	
	if args.weightedw2v:
		# Build tf-idf matrix on training documents
		sys.stdout.write("fitting tf-idf matrix on train...\n"); sys.stdout.flush()
		tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
			stop_words = ('english' if args.removeStopWords else None))
		tfidf_matrix_train =  tf.fit_transform(train['text'].astype(str))
		vocabulary = tf.vocabulary_
		sys.stdout.write("tf-idf matrix train %s\n" % str(tfidf_matrix_train.shape)); sys.stdout.flush()
		sys.stdout.write("averaging word embeddings in training data...\n"); sys.stdout.flush()
		trainDataVecs = getAvgFeatureVecs(train_words, model, num_features,
			weights = tfidf_matrix_train, word_index = vocabulary)
		trainDataVecs.dump(file_train_out)
		sys.stdout.write("writing train embeddings to %s\n" % file_train_out); sys.stdout.flush()
		
		# Build tf-idf matrix on testing documents
		sys.stdout.write("fitting tf-idf matrix on test...\n"); sys.stdout.flush()
		tf = TfidfVectorizer(analyzer='word',
			stop_words = ('english' if args.removeStopWords else None))
		tfidf_matrix_test =  tf.fit_transform(test['text'].astype(str))
		vocabulary = tf.vocabulary_
		sys.stdout.write("tf-idf matrix test %s\n" % str(tfidf_matrix_test.shape)); sys.stdout.flush()
		sys.stdout.write("averaging word embeddings in testing data...\n"); sys.stdout.flush()
		testDataVecs = getAvgFeatureVecs(test_words, model, num_features,
			weights = tfidf_matrix_test, word_index = vocabulary)
		testDataVecs.dump(file_test_out)
		sys.stdout.write("writing test embeddings to %s\n" % file_test_out); sys.stdout.flush()
		
		return trainDataVecs, testDataVecs
		
	else:
		sys.stdout.write("averaging word embeddings in training data...\n"); sys.stdout.flush()
		trainDataVecs = getAvgFeatureVecs(train_words, model, num_features)
		# write the word embeddings to file so we can read in quickly
		trainDataVecs.dump(file_train_out)
		sys.stdout.write("writing train embeddings to %s\n" % file_train_out); sys.stdout.flush()
		
		sys.stdout.write("averaging word embeddings in test data...\n"); sys.stdout.flush()
		testDataVecs = getAvgFeatureVecs(test_words, model, num_features)
		testDataVecs.dump(file_test_out)
		sys.stdout.write("writing test embeddings to %s\n" % file_test_out); sys.stdout.flush()
		
		return trainDataVecs, testDataVecs

def main():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	
	parser = argparse.ArgumentParser(description = 'Get word2vec model path')
	parser.add_argument('-w2v', dest = 'w2vpath', help = 'location of pre-built word2vec model')
	parser.add_argument('-train', dest = 'trainpath', help = 'location of pre-split training data')
	parser.add_argument('-test', dest = 'testpath', help = 'location of pre-split test data')
	parser.add_argument('-data', dest = 'datapath', help = 'location of unsplit data file')
	parser.add_argument('-size', dest = 'numSamples', 
		help = 'how many samples to use in the training', type = int)
	parser.add_argument('-loadembeddings', dest = 'loadW2Vembeddings', help = 'load stored word embeddings',
		action = 'store_true')
	parser.add_argument('-storedvecpath', dest = 'avgVecPath', help = 'location of stored word embeddings')
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
		avgVecPath = os.path.join(google_drive, 'data/'),
		splitdata = False, weightedw2v = False, removeStopWords = False, loadW2Vembeddings = False, size = 0)
	args = parser.parse_args()
	datapath = os.path.abspath(args.datapath)
	trainpath = os.path.abspath(args.trainpath)
	testpath = os.path.abspath(args.testpath)
	w2vpath = os.path.abspath(args.w2vpath)
	
	storedpath = os.path.abspath(args.avgVecPath)
	storedpath_train = os.path.join(storedpath, 'train_word_embeddings.pickle')
	storedpath_test = os.path.join(storedpath, 'test_word_embeddings.pickle')
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	logging.info('reading data')
	if args.splitdata:
		train, test = write_training(datapath, trainpath, testpath)
	else:
		train = pd.read_csv(trainpath, 
			sep = '\t', header=None, names = ['label', 'score', 'text'])
		test = pd.read_csv(testpath, 
			sep = '\t', header=None, names = ['label', 'score', 'text'])
	
	if args.loadW2Vembeddings:
		logging.info('loading word vectors')
		trainDataVecs = np.load(storedpath_train)
		testDataVecs = np.load(storedpath_test)
		sys.stdout.write("%d training posts, %d features\n" % (len(trainDataVecs), len(trainDataVecs[0])))
		sys.stdout.write("%d test posts, %d features\n" % (len(testDataVecs), len(testDataVecs[0])))
		sys.stdout.flush()
		
	else:
		logging.info('building word vectors')
		trainDataVecs, testDataVecs = computeAverage(args, train, test, 
				w2vpath, storedpath_train, storedpath_test)
		sys.stdout.write("%d training posts, %d features\n" % (len(trainDataVecs), len(trainDataVecs[0])))
		sys.stdout.write("%d test posts, %d features\n" % (len(testDataVecs), len(testDataVecs[0])))
		sys.stdout.flush()
	
	sys.stdout.write("fitting baseline model on averaged word embeddings...\n"); sys.stdout.flush()
	outputDirectory = os.path.dirname(w2vpath)
	logging.info('logistic regression')
	logitWord2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory)
	logging.info('svm with sgd')
	svmWord2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory, 10., 10.)

if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()