import pandas as pd
import numpy as np
from TrainTest import Split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import random, sys, time


def baseline(path):
	train, test = Split(path)

	# encode labels
	le = LabelEncoder()
	le.fit(train.label)
	train['y'] = le.transform(train.label)
	test['y'] = le.transform(test.label)
	
	# dense matrix
	count_vect = CountVectorizer(min_df=5)
	X_train_counts = count_vect.fit_transform(train.text.tolist())
	print 'The shape of the dense matrix: ', X_train_counts.shape

	# train model
	logit = LogisticRegression()
	model = logit.fit(X_train_counts, train.y.values)

	test_matrix = count_vect.transform(test.text.tolist())
	print 'Test sample score: %s' % str(model.score(test_matrix, test.y.values))
	print 'In sample scores: %s' % str(model.score(X_train_counts, train.y.values))
	
	# pd.Dataframe(model.coef_).to_csv('coef_.csv', index=False)
	pd.DataFrame(model.predict_proba(test_matrix)).to_csv('predict_proba.csv', index=False)

	print 'CLASSES', le.classes_ 


if __name__ == '__main__':
	print 'start'
	stime = time.time()
	script, path = sys.argv
	baseline(path)
	print 'done!'
	etime = time.time()
	ttime = etime - stime
	print ttime / 60

