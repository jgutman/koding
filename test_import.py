import numpy, scipy, pandas, sklearn, matplotlib, gensim
import os, sys, argparse, time, logging
matplotlib.use('Agg')
from matplotlib import pyplot as plt

modules = [numpy, scipy, pandas, sklearn, matplotlib, gensim]
for pkg in modules:
    print pkg.__name__, pkg.__version__