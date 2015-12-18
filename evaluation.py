import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import sys, time, os, argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def readProbability(pathToFile, header = True, index = True, svm = False, sep=',', 
        datapath = None, outpath = None) : 
    data = None
    if (type(datapath) != type(None)):
        data = pd.read_csv(datapath, sep = '\t', header = None, names = ['label', 'score', 'text'])

    labels = []
    if (type(data) != type(None)):
        le = LabelEncoder()
        le.fit(data.label)
        labels = list(le.classes_)
    
    predictions = pd.read_csv(pathToFile, header = 0 if header else None, 
        index_col = 0 if index else None, sep = sep)
    if (type(data) != type(None)):
        labels.extend(['true label', 'predicted label'])
    predictions.columns = labels
    y_true = predictions['true label']
    y_pred = predictions['predicted label']
    if not svm:
        y_prob = predictions.iloc[:, :5].as_matrix()
        y_prob = np.nan_to_num(y_prob)
    target_names = labels[:5]
    
    accuracy = np.nan
    jaccard = np.nan
    precision = np.nan
    cross_entropy = np.nan
    
    report = metrics.classification_report(y_true, y_pred, target_names=target_names)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    jaccard = metrics.jaccard_similarity_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average = "macro")
    if not svm:
        cross_entropy = metrics.log_loss(y_true, y_prob)
    
    sys.stdout.write(" Accuracy: %0.3f\n Precision: %0.3f\n Jaccard: %0.3f\n Cross-Entropy: %0.3f\n" % 
        (accuracy, precision, jaccard, cross_entropy))
    sys.stdout.write(report); sys.stdout.flush()
    confusion = metrics.confusion_matrix(y_true, y_pred)
    cm_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, target_names, outpath = outpath, title = 'Predicted Subreddits')
    
    
def plot_confusion_matrix(cm, target_names, outpath, title='Confusion matrix', cmap=plt.cm.Blues):
    # plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')   
    if (type(outpath) != type(None)):
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()
        
def main():
    root_directory = os.path.abspath('/home/cusp/rn1041/snlp/reddit/nn_reddit/')
    parser = argparse.ArgumentParser(description = 'Get predict_proba csv path')
    parser.add_argument('-data', dest = 'datapath', help = 'location of data3.txt')
    parser.add_argument('-saveoutput', dest = 'outpath', help = 'location to write plots')
    parser.add_argument('-model', dest = 'modelpath', help = 'path of model probability vectors')
    parser.add_argument('-header', dest = 'hasheader', help = 'probability file has header row',
        action = 'store_true')
    parser.add_argument('-svm', dest = 'svmmodel', help = 'is this an svm decision function file',
        action = 'store_true')
    parser.add_argument('-index', dest = 'index', help = 'file has index column')
    parser.add_argument('-sep', dest = 'sep', help = 'column delimiter for input file')
    parser.set_defaults(datapath = os.path.join(root_directory, 'data/data3.txt'),
        modelpath = os.path.join(root_directory, 'bayes_baseline/predict_proba_NB_baseline_ngram-1.csv'),
        hasheader = False, index = False, svmmodel = False, sep = ",", 
        outpath = os.path.join(root_directory, 'confusion_plots/bayes_NB_ngram_1_confusion.png'))
    args = parser.parse_args()
    
    readProbability(args.modelpath, header = args.hasheader, index = args.index, svm = args.svmmodel, 
        sep = args.sep, datapath = args.datapath, outpath = args.outpath)

if __name__ == '__main__':
    sys.stdout.write("start!\n"); sys.stdout.flush()
    stime = time.time()
    main()
    sys.stdout.write("done!\n"); sys.stdout.flush()
    etime = time.time()
    lapse = etime - stime
    sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()