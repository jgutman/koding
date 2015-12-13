import numpy as np
import pandas as pd
import glob, warnings
warnings.filterwarnings('ignore')

def RemoveList(path):
    removeList = []
    with open(path) as f:
        for i in f:
            removeList.append(i.strip())
    # add a couple of heuristic
    removeList = ['/r/', ''] + removeList
    return removeList

def CleanIt(rr):
    path ='/Users/richardnam/Google Drive/nyuClasses/SNLP/local project folder/subreddit_files'
    filenames = glob.glob(path + "/*.csv")
    outfile = open('data.txt', 'w')
    for doc in filenames:
        try:
            df = pd.read_csv(doc)
            score = df.score.values
            label = df.subreddit.values
            body = df.body.values
            skip_count = 0
            print_count = 0
            for index, string in enumerate(body):
                try:
                    if string and ("http" in string):
                        continue
                    elif string and ("[deleted]" in string):
                        continue
                    elif string:
                        tempString = []
                        row = [i.strip() for i in string.split(" ")]
                        for word in row:
                            if word not in rr:
                                tempString.append(word)
                        string = ' '.join([str(i) for i in tempString]).replace('\n', " ")
                    else:
                        continue
                except:
                    skip_count += 1
                if score[index] and label[index] and string :
                    outfile.write('\t'.join([str(label[index]), str(score[index]), str(string)]) + "\n")
                    print_count += 1
            print label[0], "skip:", skip_count, "printed:", print_count
        except:
            print "Dropped:", doc
    outfile.close()

if __name__ == '__main__':
    til = '/Users/richardnam/Google Drive/nyuClasses/SNLP/snlp project/til.txt'
    rr = RemoveList(til)
    gg = CleanIt(rr)



