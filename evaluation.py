import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def readProbability(pathToFile, header = True):
	

def main():
	google_drive = os.path.abspath('../../Google Drive/gdrive/')
	parser = argparse.ArgumentParser(description = 'Get predict_proba path')
	parser.add_argument('-model', dest = 'modelpath', help = 'path of model probability vectors')
	parser.add_argument('-header', dest = 'hasheader', help = 'probability file has header row',
		action = 'store_true')
	parser.set_defaults(modelpath = os.path.join(google_drive, 'baseline/predict_proba.csv'),
		hasheader = False)
	args = parser.parse_args()
	
	prob = readProbability(args.modelpath, args.hasheader)

if __name__ == '__main__':
	print 'start'
	stime = time.time()
	main()
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime / 60