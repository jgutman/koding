import sys, time, os, argparse
import logging, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from numpy.random import RandomState
from gensim import models
from gensim.models import doc2vec

from sklearn.utils import shuffle
from SVMtrain import SVMtrain as svm

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

def trainValidationTest( data, test_size, seed = 100 ):
	groups = data.label.unique()
	random_state = RandomState(seed = seed)
	
	# training and test
	sys.stdout.write("Building test set\n"); sys.stdout.flush()
	train = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	test = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	for i in groups:
		subtrain, subtest = train_test_split(data[data.label == i], 
			test_size = test_size, random_state = seed)
		train = train.append(subtrain)
		test = test.append(subtest)
	
	# subtraining and validation
	sys.stdout.write("Building validation set\n"); sys.stdout.flush()
	subtrain = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	val = pd.DataFrame(data = None, columns = ['label', 'score', 'text']) 
	for i in groups:
		subtrain_i, val_i = train_test_split(train[train.label == i], 
			test_size = test_size, random_state = seed)
		subtrain = subtrain.append(subtrain_i)
		val = val.append(val_i)
	
	# shuffle data	
	sys.stdout.write("Shuffling data\n"); sys.stdout.flush()
	order_train = random_state.permutation(subtrain.index)
	order_val = random_state.permutation(val.index)
	order_test = random_state.permutation(test.index)
	
	train = subtrain.loc[order_train]
	val = val.loc[order_val]
	test = test.loc[order_test]
	# train.reset_index(inplace=True, drop=True)
	# val.reset_index(inplace=True, drop=True)
	# test.reset_index(inplace=True, drop=True)
	
	sys.stdout.write("Train:\n%s\n" % train.label.value_counts())
	sys.stdout.write("Validation:\n%s\n" % val.label.value_counts())
	sys.stdout.write("Test:\n%s\n" % test.label.value_counts())
	sys.stdout.flush()
	
	return train, val, test

def traind2v( data, context, dims, d2vpath, tokenized , cores = 4, epochs = 10, seed = 150):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	documents = doc2vec.TaggedLineDocument(tokenized)
	doc_list = [doc for doc in documents]
	tag_list = [doc.tags[0] for doc in documents]

	# instantiate DM and DBOW models
	model_dbow = models.Doc2Vec( size=dims, window=context, dm=0, min_count=10, workers=cores,
		negative=10 )
	model_dm = models.Doc2Vec( size=dims, window=context, dm=1, min_count=10, workers=cores,
		negative=10 )
	
	# build vocab over all documents
	sys.stdout.write("Building vocabulary across all data\n"); sys.stdout.flush()
	model_dm.build_vocab(documents)
	model_dbow.reset_from(model_dm)
	sys.stdout.write("Vocabularies built. DM: %d words, DBOW: %d words\n" %
		(len(model_dm.vocab), len(model_dbow.vocab))); sys.stdout.flush()
	
	stime = time.time()
	for epoch in range(epochs):
		lapse = time.time() - stime
		sys.stdout.write("Training doc2vec epoch %d train (%0.0f min, %0.0f sec)\n" % 
						 (epoch+1, lapse / 60., lapse % 60.)); sys.stdout.flush()
		shuffled_documents = shuffle(doc_list, random_state = seed)
		seed += 1
		logging.info('Training DM model')
		model_dm.train(shuffled_documents)
		logging.info('Training DBOW model')
		model_dbow.train(shuffled_documents)
	
	# combine models
	model = ConcatenatedDoc2Vec([model_dm, model_dbow])
	np_model = np.vstack((model.docvecs[tag] for tag in tag_list))
	
	# save embeddings to disk
	np_model.dump(d2vpath)
	sys.stdout.write("All done. Vectors saved to %s\n" % d2vpath); sys.stdout.flush()
	return model

def tokenize_text( data, filename, tolowercase = True ):
	space = ' '
	pattern = re.compile("\W")
	output = open(filename, 'w')
	
	sys.stdout.write("Tokenizing %d documents\n" % data.shape[0]); sys.stdout.flush()
	for row in data.itertuples():
		if (row[0] % 100000. == 0):
			sys.stdout.write("Document %d\n" % row[0]); sys.stdout.flush()
		text = str(row[3]).strip()
		text = space.join(re.split(pattern, text))
		if tolowercase:
			text = text.lower()
		# data.loc[row[0], 'text'] = text # change value in dataframe, not on copy of slice
		output.write("%s\n" % text)
	
	output.close()
	sys.stdout.write("Documents tokenized to %s\n" % filename); sys.stdout.flush()

def modifyDataFromFile( data, filename ):
	newtext = open(filename, 'r')
	docs = newtext.readlines()
	data.text = [line.rstrip() for line in docs]

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-context", dest = "context", type = int)
	parser.add_argument("-dims", dest = "dims", type = int)
	parser.add_argument("-root", dest = "root_dir")
	parser.add_argument("-data", dest = "data_path")
	parser.add_argument("-d2v", dest = "store_d2v")
	parser.add_argument("-out", dest = "store_out")
	parser.add_argument("-sentences", dest = "tokenized_text")
	parser.add_argument("-subsets", dest = "subset_path")
	parser.add_argument("-tokenized", dest = "pre_tokenized", action = "store_true")
	parser.add_argument("-loadsplit", dest = "load_split_data", action = "store_true")
	parser.add_argument("-cores", dest = "cores", type = int)
	parser.add_argument("-epochs", dest = "epochs", type = int)
	
	parser.set_defaults(context = 5, dims = 100, cores = 4, epochs = 10,
		pre_tokenized = False, load_split_data = False,
		root_dir = "/home/cusp/rn1041/snlp/reddit/nn_reddit",
		data_path = "data/data3.txt",
		store_d2v = "d2vtune/embeddings/",
		store_out = "d2vtune/predictions/",
		tokenized_text = "d2vtune/tokenized.txt",
		subset_path = "d2vtune/d2v_data/")
	args = parser.parse_args()
	return args

class argdict:
	def __init__(self, context=5, dims=100):
		self.context = context
		self.dims = dims
		self.cores = 4
		self.epochs = 10
		self.root_dir = "/home/cusp/rn1041/snlp/reddit/nn_reddit"
		self.data_path = "data/data3.txt"
		self.store_d2v = "d2vtune/embeddings/"
		self.store_out = "d2vtune/predictions/"
		self.tokenized_text = "d2vtune/tokenized.txt"
		self.subset_path = "d2vtune/d2v_data/"
		self.pre_tokenized = True
		self.load_split_data = True

def main():
	args = parseArgs()
	# args = argdict()
	
	data_path = os.path.join(os.path.abspath(args.root_dir), args.data_path)
	store_d2v = os.path.join(os.path.abspath(args.root_dir), args.store_d2v)
	store_out = os.path.join(os.path.abspath(args.root_dir), args.store_out)
	tokenized_path = os.path.join(os.path.abspath(args.root_dir), args.tokenized_text)
	subset_path = os.path.join(os.path.abspath(args.root_dir), args.subset_path)
	
	# parse > tokenize > split > shuffle data
	sys.stdout.write("Reading, splitting, and shuffling the data\n"); sys.stdout.flush()
	data = pd.read_csv( data_path, sep = '\t', header = None, 
			names = ['label', 'score', 'text'] ).dropna()
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	if not args.pre_tokenized:
		logging.info('tokenizing')
		tokenize_text( data, tokenized_path )
		filename = os.path.join(subset_path, 'data.txt')
		sys.stdout.write("Writing tokenized dataframe to file %s\n" % filename)
		modifyDataFromFile( data, tokenized_path)
		data.to_csv(os.path.join(subset_path, 'data.txt'), sep = '\t', header = False, index = False)
	
	if args.load_split_data:
		# load in pre-written train, test, and validation sets from file
		train = pd.read_csv( os.path.join( subset_path, 'train.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
		val = pd.read_csv( os.path.join( subset_path, 'val.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
		test = pd.read_csv( os.path.join( subset_path, 'test.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
		sys.stdout.write("loaded data\n train: %s\n val: %s\n test: %s\n" % 
			(str(train.shape), str(val.shape), str(test.shape)))
	else:
		# split data into train, test, and validation stratfied by class
		# write datasets to file, keep index column
		train, val, test = trainValidationTest(data, test_size = .15)
		sys.stdout.write("Writing tokenized dataframe to file %s\n" % filename)
		train.to_csv( os.path.join(subset_path, 'train.txt'), sep = '\t', header = False, index = True )
		val.to_csv( os.path.join(subset_path, 'val.txt'), sep = '\t', header = False, index = True )
		test.to_csv( os.path.join(subset_path, 'test.txt'), sep = '\t', header = False, index = True )
	
	# build document embeddings
	
	model = traind2v( data, args.context, args.dims, store_d2v, tokenized_path, 
		epochs = args.epochs, cores = args.cores )
	
	# train SVM on document embeddings
	# trainVecs = 
	# valVecs =
	# testVecs =
	# trainY =
	# valY =
	# testY =
	# svm()

if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()