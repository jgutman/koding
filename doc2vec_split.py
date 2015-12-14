'''
In this script a  doc2vec model is training.
First, any rows with missing values are dropped.
Second, The text for each post is unravel, and written to a file
so that there is only one sentence per line (~9 million sentences)
Next, a doc2Vec model is train and return, also the model is saved
to the current directory.
'''
import pandas as pd
import numpy as np
from gensim import models
from gensim.models import doc2vec
import sys, time, os, logging
import argparse, string, re
from nltk.corpus import stopwords

# Word2Vec model
def d2v(sentencepath, length=300, samples=10, alpha_limit=0.0001, epochs=1, output='doc2vec_model.txt'):
	sentences = doc2vec.TaggedLineDocument(sentencepath)
	model = models.Doc2Vec(sentences, size=length, min_alpha=alpha_limit, negative=samples,
		iter=epochs, min_count=10, workers=4)
	model.save(output)
	sys.stdout.write("All done. Model saved to %s\n" % output); sys.stdout.flush()
	return model

def main(trainpath, sentencepath, d2vpath):	
	sys.stdout.write("read training data...\n"); sys.stdout.flush()
	train = pd.read_csv(trainpath, sep = '\t', header = None, names = ['label', 'score', 'text'])
	sys.stdout.write("training data: %s\n" % train.shape)
	sys.stdout.write("build document embeddings...\n"); sys.stdout.flush()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = d2v(sentencepath=sentencepath, length=300, samples=10, alpha_limit=0.001, 
		epochs=5, output=os.path.join(d2vpath, 'd2v_train_only_labels.txt'))
	
if __name__ == '__main__':
	script, trainpath, sentencepath, d2vpath = sys.argv
	sys.stdout.write("start!\n"); sys.stdout.flush()
	stime = time.time()
	main(trainpath, sentencepath, d2vpath)
	sys.stdout.write("done!\n"); sys.stdout.flush()
	etime = time.time()
	lapse = etime - stime
	sys.stdout.write("%0.2f min\n" % (lapse / 60.); sys.stdout.flush()