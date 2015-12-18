cd /home/cusp/rn1041/snlp/reddit/nn_reddit

source activate py27

python -c "import pandas as pd; path = 'data/data3.txt'; data = pd.read_csv(path, sep ='\t', header = None, names = ['label', 'score', 'text']); data = data.dropna().reset_index(drop = True); data.to_csv(path, sep = '\t', header = False, index = False)"

python ../koding/build_tune_d2v.py -epochs 0

