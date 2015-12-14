'''
In this script a  word2vec model is training.
First, any rows with missing values are dropped.
Second, The text for each post is unravel, and written to a file
so that there is only one sentence per line (~9 million sentences)
Next, a Word2Vec model is train and return, also the model is saved
to the current directory.
'''
import pandas as pd
import numpy as np
from gensim import models
from gensim.models import word2vec
import sys, time, os, logging
import argparse, string, re
from nltk.corpus import stopwords

# Create a file of just sentences for NN
def Sentences(data, filename='sentences.txt', tolowercase = False):
	'''
	argument: data = Pandas dataframe, filename = file name
	returns: string of the path to sentences
	'''
	output = open(filename, 'w')
	space = ' '
	pattern = re.compile("\W")
	for row in data.itertuples():
		text = str(row[3])
		text = space.join(re.split(pattern, text))
		if tolowercase:
			text = text.lower()
		output.write(text+'\n')
	output.close()
	sys.stdout.write("returns path to file: %s\n"  % filename)
	return filename
	
# Count number of lines in a file
def file_len(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return (i+1)
 
# Word2Vec model
def w2v(sentencepath, length=300, context=5, 
		samples=10, alpha_limit=0.0001, epochs=1, 
		output='word2vec_model.txt'):
	sentences = word2vec.LineSentence(sentencepath)
	model = models.Word2Vec(sentences, size=length, window=context, 
		min_alpha=alpha_limit, negative=samples,
		iter=epochs, min_count=10, workers=4)
	model.save(output)
	sys.stdout.write("All done. Model saved to %s\n" % output); sys.stdout.flush()
	return model

def main(trainpath, w2vpath, sentencepath):
	sys.stdout.write("read training data...\n"); sys.stdout.flush()
	path = os.path.abspath(trainpath)
	train = pd.read_csv(path, sep = '\t', header = None, names = ['label', 'score', 'text'])
	sys.stdout.write("training data: %s\n" % str(train.shape))
	sys.stdout.write("write sentences to file...\n"); sys.stdout.flush()
	sentencePath = Sentences(data = train, filename = os.path.abspath(sentencepath))
	sys.stdout.write("build word embeddings...\n"); sys.stdout.flush()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = w2v(sentencepath = sentencePath, length = 300, 
		context = 5, samples = 10, alpha_limit = 0.001, 
		epochs = 1, output = os.path.abspath(w2vpath))
	
if __name__ == '__main__':
	script, trainpath, w2vpath, sentencepath = sys.argv
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main(trainpath, w2vpath, sentencepath)
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()