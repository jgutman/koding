from gensim import models
import argparse, os
import nltk

from TrainTest import Split
from TrainTest import ParseData
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import random, sys, time
    
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
		
	parser.set_defaults(google_drive = os.path.abspath('../../Google Drive/gdrive/'))
	parser.set_defaults(doc2vpath = 'doc2vec/d2v_output01.txt', 
		trainpath = 'data/train.txt', 
		testpath = 'data/test.txt',
		datapath = 'data2.txt', 
		splitdata = False, size = 0)
	args = parser.parse_args()
	trainpath = os.path.join(args.google_drive, args.trainpath)
	testpath = os.path.join(args.google_drive, args.testpath)
	datapath = os.path.join(args.google_drive, args.datapath)
	doc2vpath = os.path.join(args.google_drive, args.doc2vpath)
	
if __name__ == '__main__':
	print 'start'
	stime = time.time()
	main()
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime / 60