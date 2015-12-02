from gensim import models
import argparse, os
import nltk
from TrainTest import Split
from TrainTest import ParseData
import pandas as pd

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
	
def main():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	
	parser = argparse.ArgumentParser(description='Get word2vec model path')
	parser.add_argument('-w2v', dest = 'w2vpath', help='location of pre-built word2vec model')
	parser.add_argument('-train', dest = 'trainpath', help='location of pre-split training data')
	parser.add_argument('-test', dest = 'testpath', help='location of pre-split test data')
	parser.add_argument('-data', dest = 'datapath', help='location of unsplit data file')
	parser.add_argument('-split', dest = 'splitdata', help='split data into train and test?',
		action='store_true')
	
	parser.set_defaults(w2vpath = os.path.join(google_drive, 'w2v_output1/w2v_output01.txt'), 
		trainpath = os.path.join(google_drive, 'data/train.txt'), 
		testpath = os.path.join(google_drive, 'data/test.txt'),
		datapath = os.path.join(google_drive, 'data.txt'), splitdata = False)
	args = parser.parse_args()
	
	model = models.Word2Vec.load(args.w2vpath)
	if (args.splitdata):
		train, test = write_training(args.datapath, args.trainpath, args.testpath)
	else:
		train = pd.read_csv(args.trainpath, sep = '\t', header=None, names = ['label', 'score', 'text'])
		test = pd.read_csv(args.trainpath, sep = '\t', header=None, names = ['label', 'score', 'text'])

if __name__ == '__main__':
	main()