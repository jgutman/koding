import sys, time, os, argparse
import logging, re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from numpy.random import RandomState
from gensim import models
from gensim.models import doc2vec

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

def traind2v( train, val, test, context, dims, d2vpath, tokenized ):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	documents = doc2vec.TaggedLineDocument(tokenized)
	
	model_dbow = models.Doc2Vec( size=dims, window=context, dm=0, min_count=10, workers=4,
		negative=10, sample=1e-5 )
	model_dm = models.Doc2Vec( size=dims, window=context, dm=1, min_count=10, workers=4,
		negative=10, sample=1e-5 )
	model_dbow.build_vocab(documents)
	model_dm.build_vocab(documents)
	
	
	
	model.save(output)
	sys.stdout.write("All done. Model saved to %s\n" % output); sys.stdout.flush()
	

def tokenize_text( data, filename, tolowercase = True ):
	space = ' '
	pattern = re.compile("\W")
	output = open(filename, 'w')
	
	sys.stdout.write("Tokenizing %d documents\n" % data.shape[0]); sys.stdout.flush()
	for row in data.itertuples():
		text = str(row[3])
		text = space.join(re.split(pattern, text))
		if tolowercase:
			text = text.lower()
		data.ix['text'][row[0]] = text # change value in dataframe, not on copy of slice
		output.write(text+'\n')
	
	output.close()
	sys.stdout.write("Documents tokenized to %s\n" % filename); sys.stdout.flush()
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-context", dest = context)
	parser.add_argument("-dims", dest = dims)
	parser.add_argument("-root", dest = root_dir)
	parser.add_argument("-data", dest = data_path)
	parser.add_argument("-d2v", dest = store_d2v)
	parser.add_argument("-out", dest = store_out)
	parser.add_argument("-sentences", dest = tokenized_text)
	parser.add_argument("-subsets", dest = subset_path)
	parser.add_argument("-tokenized", dest = pre_tokenized, action = store_true)
	parser.add_argument("-loadsplit", dest = load_split_data, action = store_true)
	
	parser.set_defaults(context = "5", dims = "100", pre_tokenized = False,
		root_dir = "/home/cusp/rn1041/snlp/reddit/nn_reddit",
		data_path = "/data/data3.txt",
		store_d2v = "/d2vtune/embeddings/",
		store_out = "/d2vtune/predictions/",
		tokenized_text = "/d2vtune/tokenized.txt",
		subset_path = "/d2vtune/data/")
	args = parser.parse_args()
	
	data_path = os.path.join(os.path.abspath(args.root_dir), args.data_path)
	store_d2v = os.path.join(os.path.abspath(args.root_dir), args.store_d2v)
	store_out = os.path.join(os.path.abspath(args.root_dir), args.store_out)
	tokenized_path = os.path.join(os.path.abspath(args.root_dir), args.tokenized_text)
	subset_path = os.path.join(os.path.abspath(args.root_dir), args.subset_path)
	
	context = int(args.context)
	dims = int(args.context)
	
	# parse tokenize split shuffle data
	sys.stdout.write("Reading, splitting, and shuffling the data\n"); sys.stdout.flush()
	data = pd.read_csv( data_path, sep = '\t', header = None, 
			names = ['label', 'score', 'text'] ).dropna()
			
	if not args.pre_tokenized:
		tokenize_text( data, tokenized_path )
		data.to_csv(os.path.join(subset_path, 'data.txt'), sep = '\t', header = False, index = False)
		
	if args.load_split_data:
		# load in pre-written train, test, and validation sets from file
		train = pd.read_csv( os.path.join( subset_path, 'train.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
		val = pd.read_csv( os.path.join( subset_path, 'val.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
		test = pd.read_csv( os.path.join( subset_path, 'test.txt'), sep = '\t', header = None,
			names = ['label', 'score', 'text'], index_col = 0 )
	else:
		# split data into train, test, and validation stratfied by class
		# write datasets to file, keep index column
		train, val, test = trainValidationTest(data, test_size = .15)
		train.to_csv( os.path.join(subset_path, 'train.txt'), sep = '\t', header = False, index = True )
		val.to_csv( os.path.join(subset_path, 'val.txt'), sep = '\t', header = False, index = True )
		test.to_csv( os.path.join(subset_path, 'test.txt'), sep = '\t', header = False, index = True )
	
	# build document embeddings
	
	traind2v( train, val, test, context, dims, store_d2v, tokenized = tokenized_path )
	
	# train SVM on document embeddings
	

if __name__ == '__main__':
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main()
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()