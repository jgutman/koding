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
from gensim.models import word2vec
import sys


# Parse cleaned data into a vector
def ParseData(path):
    '''
    arguments: path = path to data.txt
    returns: an array of data
    '''
    tempD = []
    skip_count = 0
    with open(path) as f:
        try:
            for line in f:
                row = line.split('\t')
                label = row[0]
                score = row[1]
                text = ' '.join(row[2:]).replace("\t", " ").strip()
                if label != 'nan':
                    tempD.append([label, score, text])
        except:
            skip_count +=1
    sys.stdout.write('\rerrors catched: %d' % skip_count)
    return tempD

# Create a file of just sentences for NN, break at '.'
def Sentences(data, filename='sentences'):
    '''
    arguement: data = array of data, filename = file name
    returns: string of the path to sentences
    '''
    output = open(filename+'.txt', 'w')
    for row in data:
        text = row[2]
        sentences = text.split('.')
        for line in sentences:
            if str(line).strip():
                output.write(str(line).strip()+"\n")
    output.close()
    sys.stdout.write('\rreturns path to file: ' + filename +'.txt')
    return filename+'.txt'

# Word2Vec model
def w2v(sentencepath, length=300, context=5, 
        samples=10, alpha_limit=0.0001, epochs=1, output='word2vec_model.txt'):
    sentences = word2vec.LineSentence(sentencePath)
    model = word2vec.Word2Vec(sentences, size=length, window=context, 
                              min_alpha=alpha_limit, negative=samples,
                              iter=epochs, min_count=100, workers=4)
    model.save(output)
    sys.stdout.write("\rAll done. Model saved to " + output)
    return model


if __name__ == '__main__':
    script, trainpath = sys.argv
    train = pd.read_csv(trainpath, sep = '\t', header=None, 
			names = ['label', 'score', 'text'])
    sentencePath = Sentences(data=trin, filename='train_w2v_ready')
    model = w2v(sentencepath=sentencePath, length=300, context=5, 
                samples=10, alpha_limit=0.001, epochs = 1, output='w2v_train_only.txt')
