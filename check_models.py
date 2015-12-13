from gensim import models
import sys

def main(w2vpath, d2vpath):
	w2vmodel = models.Word2Vec.load(w2vpath)
	d2vmodel = models.Doc2Vec.load(d2vpath)
	
	print w2vmodel.syn0.shape
	print d2vmodel.syn0.shape
	print len(d2vmodel.docvecs)

if __name__ == '__main__':
	script, w2vpath, d2vpath = sys.argv
	main(w2vpath, d2vpath)