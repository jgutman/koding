from gensim import models
import argparse, os
import nltk

from TrainTest import Split
from TrainTest import ParseData
import pandas as pd
import numpy as np
from baseline_word2vec import write_training

from sklearn.preprocessing import LabelEncoder
import random, sys, time
from baseline_word2vec import docWordList
from nltk.corpus import stopwords

def logitDoc2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
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

	outfile = os.path.join(outputPath, 'doc2vec_logit_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	print 'CLASSES', le.classes_ 

def svmWord2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
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

	outfile = os.path.join(outputPath, 'doc2vec_svm_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	print 'CLASSES', le.classes_ 

def getTestVectors(test, model, remove_stopwords = False):
	# test should be a pandas dataframe, column 'text' contains the text of the document
	# model should be a pre-trained Doc2Vec model
	
	# Preallocate a 2D numpy array, for speed
	num_features = int(model.syn0.shape[1])
	docFeatureVecs = np.zeros((len(test.index), num_features), dtype=np.float32)
	
	for post in test.itertuples(): 
		text = str(post[3])
		wordList = docWordList(text, remove_stopwords)
		docVector = model.infer_vector(wordList)
		docFeatureVecs[post[0]] = docVector 
	return docFeatureVecs

def main():
	parser = argparse.ArgumentParser(description = 'Get doc2vec model path')
	parser.add_argument('-google', dest = 'google_drive', help = 'location of gdrive path')
	parser.add_argument('-doc2v', dest = 'doc2vpath', help = 'location of pre-built doc2vec model')
	parser.add_argument('-train', dest = 'trainpath', help = 'location of pre-split training data')
	parser.add_argument('-test', dest = 'testpath', help = 'location of pre-split test data')
	parser.add_argument('-data', dest = 'datapath', help = 'location of unsplit data file')
	parser.add_argument('-size', dest = 'numSamples', 
		help = 'how many samples to use in the training', type = int)
	parser.add_argument('-split', dest = 'splitdata', 
		help = 'split data into train and test?', action = 'store_true')
	parser.add_argument('-stopwords',  dest = 'removeStopWords', 
		help = 'remove English stop words', action = 'store_true')
		
	parser.set_defaults(google_drive = os.path.abspath('../../Google Drive/gdrive/'))
	parser.set_defaults(doc2vpath = 'doc2vec/d2v_train_only_labels.txt', 
		trainpath = 'data/train2.txt', 
		testpath = 'data/test2.txt',
		datapath = 'data3.txt', 
		splitdata = False, removeStopWords = False, size = 0)
	args = parser.parse_args()
	trainpath = os.path.join(os.path.abspath(args.google_drive), args.trainpath)
	testpath = os.path.join(os.path.abspath(args.google_drive), args.testpath)
	datapath = os.path.join(os.path.abspath(args.google_drive), args.datapath)
	doc2vpath = os.path.join(os.path.abspath(args.google_drive), args.doc2vpath)
	
	print "loading doc2vec..."
	model = models.Doc2Vec.load(doc2vpath)
	print "loading train and test data..."
	if args.splitdata:
		train, test = write_training(datapath, trainpath, testpath)
	else:
		train = pd.read_csv(trainpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
		test = pd.read_csv(testpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
	
	print "fetching training document embeddings..."
	trainDataVecs = model.docvecs
	print len(trainDataVecs), len(trainDataVecs[0])
	
	print "inferring test document embeddings..."
	testDataVecs = getTestVectors(test, model, remove_stopwords = args.removeStopWords)
	np.savetxt((testpath+'.d2v.embeddings.txt'), testDataVecs, delimiter='\t')
	print len(testDataVecs), len(testDataVecs[0])
	
	print "fitting logit and svm model on document embeddings..."
	outputDirectory = os.path.dirname(doc2vpath)
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