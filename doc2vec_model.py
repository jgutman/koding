from gensim import models
import argparse, os, logging
import nltk

from TrainTest2 import Split
from TrainTest2 import ParseData
import pandas as pd
import numpy as np
from baseline_word2vec import write_training

from sklearn.preprocessing import LabelEncoder
import random, sys, time
from baseline_word2vec import docWordList
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn import svm

def logitDoc2Vec(train, test, trainDataVecs, testDataVecs, outputPath):
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

	outfile = os.path.join(outputPath, 'doc2vec_logit_predict_proba.csv')
	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv(outfile, 
		sep = '\t', header = list(le.classes_), index = False)
	
	sys.stdout.write('CLASSES: %s\n' % le.classes_)
	sys.stdout.flush()

def svmDoc2Vec(train, test, trainDataVecs, testDataVecs, outputPath, lamb, zoom, 
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
	upper = 10
	for level in xrange(zoom):
		lambda_range = np.linspace(lower, upper, lamb)
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
		sys.stdout.write('best: %0.4f score: %0.4f\n'  % (best, nested_scores[best])
		sys.stdout.flush()
	clf = SGDClassifier(alpha=lambda_range[best], loss='hinge', penalty='l2', 
				l1_ratio=0, n_iter=5, n_jobs=4, shuffle=True,  
				model = clf.fit(subtrain_X, subtrain_Y)
	df = pd.DataFrame(model.decision_function(testDataVecs), 
				columns=[v for i,v in enumerate(le_classes_)])
	df['y'] = test.y.values
	df['predict'] = model.predict(testDataVecs)
	df.to_csv('decision_function_svm_doc2vec.csv', sep='\t', index=False)
	sys.stdout.write('FINAL SCORE %0.4f\n' % model.score(testDataVecs, test.y.values))
	sys.stdout.flush()

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
	parser.add_argument('-loadtest', dest = 'loadTestVecs', help = 'load test document embeddings',
		action = 'store_true')
	parser.add_argument('-testvecpath', dest = 'testVecPath', help = 'location of stored test embeddings')
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
		testVecPath = 'doc2vec/test2.d2v.embeddings.pickle',
		splitdata = False, removeStopWords = False, loadTestVecs = False, size = 0)
	args = parser.parse_args()
	trainpath = os.path.join(os.path.abspath(args.google_drive), args.trainpath)
	testpath = os.path.join(os.path.abspath(args.google_drive), args.testpath)
	datapath = os.path.join(os.path.abspath(args.google_drive), args.datapath)
	doc2vpath = os.path.join(os.path.abspath(args.google_drive), args.doc2vpath)
	testVecPath = os.path.join(os.path.abspath(args.google_drive), args.testVecPath)
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sys.stdout.write("loading doc2vec...\n"); sys.stdout.flush()
	model = models.Doc2Vec.load(doc2vpath)
	sys.stdout.write("loading train and test data..."); sys.stdout.flush()
	if args.splitdata:
		train, test = write_training(datapath, trainpath, testpath)
	else:
		train = pd.read_csv(trainpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
		test = pd.read_csv(testpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
	
	sys.stdout.write("fetching training document embeddings...\n"); sys.stdout.flush()
	trainDataVecs = model.docvecs
	sys.stdout.write("%d training posts, %d features\n" % (len(trainDataVecs), len(trainDataVecs[0]))
	
	sys.stdout.write("inferring test document embeddings...\n"); sys.stdout.flush()
	if args.loadTestVecs:
		testDataVecs = np.load(testVecPath)
		sys.stdout.write("%d test posts, %d features\n" % (len(testDataVecs), len(testDataVecs[0]))
		sys.stdout.flush()
	else:	
		testDataVecs = getTestVectors(test, model, remove_stopwords = args.removeStopWords)
		testDataVecs.dump(testVecPath)
		sys.stdout.write("%d test posts, %d features\n" % (len(testDataVecs), len(testDataVecs[0]))
		sys.stdout.flush()
	
	sys.stdout.write("fitting logit and svm model on document embeddings...\n"); sys.stdout.flush()
	outputDirectory = os.path.dirname(doc2vpath)
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	logitDoc2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory)
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	svmDoc2Vec(train, test, trainDataVecs, testDataVecs, outputDirectory)
	
if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.); sys.stdout.flush()