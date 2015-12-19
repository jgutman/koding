from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, time, os, argparse, logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy.random import RandomState
from sklearn.utils import shuffle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-root", dest = "rootdir")
    parser.add_argument("-data", dest = "datapath")
    parser.add_argument("-embeddings", dest = "embedpath")
    parser.add_argument("-out", dest = "outpath")
    parser.set_defaults(
        rootdir = "/scratch/jg3862/gdrive/", 
        # rootdir = "../../Google Drive/gdrive",
        datapath = "d2vtune/d2v_data/data.txt", 
        embedpath = "d2vtune/embeddings/d2v_context_10_dim_100_dm_dbow.pickle",
        outpath = "tsne_project_d2v_context_10_dim_100_dm_dbow.png")
    args = parser.parse_args()
    datapath = os.path.join(os.path.abspath(args.rootdir), args.datapath)
    embedpath = os.path.join(os.path.abspath(args.rootdir), args.embedpath)
    outpath = os.path.join(os.path.abspath(args.rootdir), args.outpath)
    
    # Parse labeled input data
    sys.stdout.write("Parsing data from %s\n" % datapath); sys.stdout.flush()
    data = pd.read_csv(datapath, sep='\t', header = None, 
        names = ['label', 'score', 'text']).dropna().reset_index(drop=True)
    le = LabelEncoder()
    le.fit(data.label)
    data['y'] = le.transform(data.label)
    select = subsample(data, le)
    plot_and_cluster(data.y.loc[select], embedpath, outpath)

def subsample(data, le, n_samples = 20000, seed = 1000):
    random = RandomState(seed)
    
    sys.stdout.write("Subsampling %d members from each of %d classes\n" %
        (n_samples, len(le.classes_)))
    indices = []
    for i, label in enumerate(le.classes_):
        class_index = list(data.index[data.y == i])
        subsample = shuffle(class_index, random_state = random, n_samples = n_samples)
        indices.extend(subsample)
    sys.stdout.write("Done. %d samples chosen.\n" % len(indices))
    return indices

def plot_and_cluster(labels, embedpath, outpath, reduced_dims = 2):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    ts = TSNE(reduced_dims, learning_rate = 100)
    sys.stdout.write("Reading embeddings from %s\n" % embedpath); sys.stdout.flush()
    dataVecs = np.load(embedpath)
    sys.stdout.write("Embedded data size: %s\n" % str(dataVecs.shape)); sys.stdout.flush()
    sampleDataVecs = dataVecs[labels.index, :]
    sys.stdout.write("Sample data size: %s\n" % str(sampleDataVecs.shape)); sys.stdout.flush()
    
    sys.stdout.write("Reducing vectors of dimension %d to dimension %d\n" %
        (sampleDataVecs.shape[1], reduced_dims)); sys.stdout.flush()
    logging.info("Fitting TSNE")
    reducedVecs = ts.fit_transform(sampleDataVecs)
    logging.info("Done fitting TSNE")
    
    # Color points by document label to see if Word2Vec/Doc2Vec can separate them
    # plt.figure()
    sys.stdout.write("Plotting TSNE projections\n"); sys.stdout.flush()
    for i in range(len(labels)):
        if (i % 10000. == 0):
            sys.stdout.write("Plotting document %d of %d\n" % 
                (i+1, len(labels))); sys.stdout.flush()
        label = labels.iloc[i]
        if label == 0:
            color = 'b'
        elif label == 1:
            color = 'r'
        elif label == 2:
            color = 'c'
        elif label == 3:
            color = 'm'
        else:
            color = 'g'
        plt.plot(reducedVecs[i, 0], reducedVecs[i, 1], marker='o', color=color, markersize=8)
    plt.savefig(outpath)
    plt.close()

if __name__ == '__main__':
    sys.stdout.write("start!\n"); sys.stdout.flush()
    stime = time.time()
    main()
    sys.stdout.write("done!\n"); sys.stdout.flush()
    etime = time.time()
    lapse = etime - stime
    sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()