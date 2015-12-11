import pandas as pd
import sys

def cleanData(filepath, output):
    data = pd.read_csv(filepath, sep='\t', header = None, names = ['label', 'score', 'text'], error_bad_lines=False)
    screwups = data.label.value_counts()[data.label.value_counts() < 100000] 
    weirdRows = []
    for partialPost in screwups.index:
        rowIndexer = data[data.label == partialPost].index
        for row in rowIndexer:
            weirdRows.append(row)
    weirdRows.sort()
    dataCleaned = data.copy()
    for row in weirdRows:
        baseRow = row
        while (str(dataCleaned.text.loc[baseRow]) == 'nan'):
            baseRow = baseRow - 1
        part1 = str(dataCleaned.text.loc[baseRow])
        part2 = str(dataCleaned.label.loc[row])
        dataCleaned.text.loc[baseRow] = part1 + ' ' + part2
    emptyRows = dataCleaned[pd.isnull(dataCleaned.label)].index
    dataCleaned.drop(emptyRows, inplace = True)
    textCounts = dataCleaned.text.value_counts(dropna = False)
    doubles = textCounts[textCounts.values == 2]
    duplicateRowsToDrop = []
    for post in list(doubles.index):
        if len(str(post)) < 20:
            continue
        doubleIndex =  list((dataCleaned[dataCleaned.text == post]).index)
        duplicateRowsToDrop.append(doubleIndex[1])
    dataTrimmed = dataCleaned.drop(duplicateRowsToDrop)
    dataTrimmed.to_csv(outputPath, sep = '\t', header = False, index = False)
    
if __name__ == '__main__':
    script, inputPath, outputPath = sys.argv
    cleanData(inputPath, outputPath)

