from gensim import models
import argparse, os
import nltk

from TrainTest import Split
from TrainTest import ParseData
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
import random, sys, time

def write_training(datapath, trainpath, testpath):
	train, test = Split(datapath)
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

def makeFeatureVec(words, weights = None, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,) , dtype="float32")
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
            		
	
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(documents, weights = None, model, num_features):
    # Given a collection of documents (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    
    counter = 0.
    feature_names = None
    if (weights != None):
    	feature_names = weights.get_feature_names() 
    
    # Preallocate a 2D numpy array, for speed
    docFeatureVecs = np.zeros((len(documents),num_features),dtype="float32")

	for post in documents:
       # Print a status message every 1000th review
    	if counter%1000. == 0.:
        	print "Reddit post %d of %d" % (counter, len(documents))
        	if (weights == None):
    			docFeatureVecs[counter] = makeFeatureVec(post, model, num_features)
    		else:
    			weightPost = weights[counter]
    			docFeatureVecs[counter] = makeFeatureVec(post, weightPost, model, 
    				num_features, feature_names)  		
       counter = counter + 1.
    return docFeatureVecs

def baselineWord2Vec(train, test, trainDataVecs, testDataVecs):
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

	pd.DataFrame(model.predict_proba(testDataVecs)).to_csv('word2vec_predict_proba.csv', 
		index=False)
	
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
	parser.add_argument('-split', dest = 'splitdata', help = 'split data into train and test?',
		action = 'store_true')
	parser.add_argument('-weighted', dest = 'weightedw2v', help = 'use tf-idf weighting for words',
		action = 'store_true')
	parser.add_argument('-stopwords', dest = 'removeStopWords', help = 'remove English stop words',
		action = 'store_true')
	
	parser.set_defaults(w2vpath = os.path.join(google_drive, 'w2v_output1/w2v_output01.txt'), 
		trainpath = os.path.join(google_drive, 'data/train.txt'), 
		testpath = os.path.join(google_drive, 'data/test.txt'),
		datapath = os.path.join(google_drive, 'data.txt'), 
		splitdata = False, weightedw2v = False, removeStopWords = False)
	args = parser.parse_args()
	
	model = models.Word2Vec.load(args.w2vpath)
	if args.splitdata:
		train, test = write_training(args.datapath, args.trainpath, args.testpath)
	else:
		train = pd.read_csv(args.trainpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
		test = pd.read_csv(args.trainpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
	
	word_vectors = model.syn0
	vocabulary_size = word_vectors.shape[0]
	num_features = word_vectors.shape[1]
	
	# Should we remove stopwords here or just implement the tf-idf weighting scheme?
	train_words = []
	test_words = []
	for post in train['text'].astype(str):
		train_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	for post in test['text'].astype(str):
		test_words.append(docWordList(post, remove_stopwords = args.removeStopWords))
	
	if args.weightedw2v:
		tf = TfidfVectorizer(analyzer='word', vocabulary = model.vocab.keys(),
			stop_words = ('english' if args.removeStopWords else None))
		tfidf_matrix_train =  tf.fit_transform(train['text'].astype(str))
		tfidf_matrix_test =  tf.fit_transform(test['text'].astype(str))
		
		trainDataVecs = getAvgFeatureVecs(train_words, weights = tfidf_matrix_train, 
			model = model, num_features = num_features)
		testDataVecs = getAvgFeatureVecs(test_words, weights = tfidf_matrix_test, 
			model = model, num_features = num_features)
		
		
		
	else: 
		trainDataVecs = getAvgFeatureVecs(train_words, model = model, num_features = num_features)
		testDataVecs = getAvgFeatureVecs(test_words, model = model, num_features = num_features)
	
	baselineWord2Vec(train, test, trainDataVecs, testDataVecs)

if __name__ == '__main__':
	print 'start'
	stime = time.time()
	main()
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime % 60