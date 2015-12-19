import numpy, scipy, pandas, sklearn, matplotlib, gensim, nltk
import os, sys, argparse, time, logging
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

modules = [numpy, scipy, pandas, sklearn, matplotlib, gensim, nltk]
for pkg in modules:
    print pkg.__name__, pkg.__version__