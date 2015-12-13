import pandas as pd
import sys

def cleanData(filepath, outputPath):
	data = pd.read_csv(filepath, sep='\t', header = None, names = ['label', 'score', 'text'])
	screwups = data.label.value_counts()[data.label.value_counts() < 100000] 
	
	print 'Finding cells broken by newlines...'
	weirdRows = []
	for partialPost in screwups.index:
		rowIndexer = data[data.label == partialPost].index
		for row in rowIndexer:
			weirdRows.append(row)
			
	weirdRows.sort()
	print 'Removing strange characters...'
	dataCleaned = data.copy()
	for row in dataCleaned.itertuples():
		text = str(row[1])
		text = text.replace('\r', ' ')
		text = text.replace('\n', ' ')
		text = text.replace('\t', ' ')
		text = text.replace('&gt;', ' ')
	
	for row in weirdRows:
		baseRow = row
		while (str(dataCleaned.text.loc[baseRow]) == 'nan'):
			baseRow = baseRow - 1
		part1 = str(dataCleaned.text.loc[baseRow])
		part2 = str(dataCleaned.label.loc[row])
		dataCleaned.text.loc[baseRow] = part1 + ' ' + part2
	
	print 'Merging split newline cells...'
	dataCleaned.drop(weirdRows, inplace = True)
	dataCleaned.dropna(subset = ['label', 'text'], inplace = True)
	
	print 'Dealing with suspicious duplicates...'
	textCounts = dataCleaned.text.value_counts(dropna = False)
	doubles = textCounts[textCounts.values == 2]
	
	duplicateRowsToDrop = []
	for post in list(doubles.index):
		if len(str(post)) < 20:
			continue
		doubleIndex =  list((dataCleaned[dataCleaned.text == post]).index)
		duplicateRowsToDrop.append(doubleIndex[1])
	
	print 'Removing whitespace characters...'
	dataTrimmed = dataCleaned.drop(duplicateRowsToDrop)
	dataTrimmed.dropna(subset = ['label', 'text'], inplace = True)
	print 'Copying and writing to file...'
	dataTrimmed.to_csv(outputPath, sep = '\t', header = False, index = False)

if __name__ == '__main__':
	script, inputPath, outputPath = sys.argv
	cleanData(inputPath, outputPath)
	print 'Wrote clean data to' + outputPath
