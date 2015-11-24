import numpy as np
import pandas as pd



'''
Reads in the data and return the string of text
'''
def ReadData(path):
    data = []
    with open(path) as f:
        for line in f:
            row  = line.split('\t')
            if len(row) == 22:
                data.append(row[17])
            else:
                data.append('SKIPPED')
    return data


'''
imports a list of reddit acroymns (spelled wrong)
appends a few written in hand
'''
def RemoveList(path):
    removeList = []
    with open(path) as f:
        for i in f:
             removeList.append(i.strip())
    # add a couple of heuristic
    removeList = ['/r/', ''] + removeList
    return removeList


'''
removes the acros, drops an sample with hyperlink or
that has the [deleted] string in it.
'''
def CleanStrings(sentences, removeList):
    sentences2 = []
    for line in sentences:
        tempSentences = []
        row = [k.strip() for k in line.split(' ')]
        for word in row:
            if word in removeList:
                pass
            else:
                tempSentences.append(word)
        if tempSentences and (tempSentences[0] == '[deleted]'):
            sentences2.append('SKIPPED')
        elif tempSentences and ('http' in ' '.join(tempSentences)):
            sentences2.append('SKIPPED')
        elif tempSentences:
            sentences2.append(' '.join(tempSentences))
        else:
            sentences2.append('SKIPPED')
    return sentences2    



if __name__ == '__main__':
    sentences = ReadData('sample.txt')
    rr = RemoveList('til.txt')
    gg = CleanStrings(sentences, rr)


