import os, sys, time
from evaluation import readProbability
import pandas as pd

def main():
    directory = '/Users/jacqueline/Google Drive/gdrive/w2vtune/predictions/'
    outpath = '/Users/jacqueline/Google Drive/gdrive/confusion_plots/'
    datapath = '/Users/jacqueline/Google Drive/gdrive/d2vtune/d2v_data/data.txt'
    
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print file
            confusion_path = os.path.join(outpath, file)
            confusion_path = confusion_path.replace('.csv', '.png')
            print confusion_path
            pathToFile = os.path.join(directory, file)
            readProbability(pathToFile, header = True, index = False, svm = True, 
                datapath = datapath, outpath = confusion_path)
        
if __name__ == '__main__':
    sys.stdout.write("start!\n"); sys.stdout.flush()
    stime = time.time()
    main()
    sys.stdout.write("done!\n"); sys.stdout.flush()
    etime = time.time()
    lapse = etime - stime
    sys.stdout.write("%0.2f min\n" % (lapse / 60.)); sys.stdout.flush()
