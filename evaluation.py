import pandas as pd
from sklearn import metrics

def readProbability(pathToFile, header = True):


def main():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	parser = argparse.ArgumentParser(description = 'Get predict_proba path')
	parser.add_argument('-model', dest = 'modelpath', help = 'path of model probability vectors')
	parser.set_defaults(modelpath = os.path.join(google_drive, 'baseline/predict_proba.csv'))
	args = parser.parse_args()

if __name__ == '__main__':
	print 'start'
	stime = time.time()
	main()
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime / 60