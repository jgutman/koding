from baseline_word2vec import write_training
import sys

def main(datapath, trainpath, testpath):
	write_training(datapath, trainpath, testpath)
	print "training data written to " + trainpath
	print "test data written to " + testpath

if __name__ == '__main__':
	script, datapath, trainpath, testpath = sys.argv
	main(datapath, trainpath, testpath)
