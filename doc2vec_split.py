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
from gensim.models import doc2vec
import sys, time, os, logging
  
# Word2Vec model
def d2v(sentencepath, length=300, samples=10, alpha_limit=0.0001, epochs=1, output='doc2vec_model.txt'):
    sentences = doc2vec.TaggedLineDocument(sentencepath)
    model = doc2vec.Doc2Vec(sentences, size=length, min_alpha=alpha_limit, negative=samples,
                            iter=epochs, min_count=10, workers=4)
    model.save(output)
    print "All done. Model saved to %s" % (output)
    return model

def main(trainpath, sentencepath, d2vpath):	
	print 'read training data...'
	train = pd.read_csv(trainpath, sep = '\t', header = None, names = ['label', 'score', 'text'])
	print train.shape
	print 'build document embeddings...'
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level
	model = d2v(sentencepath=sentencepath, length=300, samples=10, alpha_limit=0.001, 
		epochs=5, output=os.path.join(d2vpath, 'd2v_train_only_labels.txt'))
	
if __name__ == '__main__':
	script, trainpath, sentencepath, d2vpath = sys.argv
	print 'start!'
	start_time = time.time()
	main(trainpath, sentencepath, d2vpath)
	lapse = time.time() - start_time
	print "%0.2f min" % (lapse / 60.)
