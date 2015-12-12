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
	dataCleaned = data.copy()
	
	for row in weirdRows:
		baseRow = row
		compare_string = str(dataCleaned.text.loc[baseRow]).lower()
		while (compare_string.strip() == 'nan' or compare_string.strip() == ''):
			baseRow = baseRow - 1
			compare_string = str(dataCleaned.text.loc[baseRow]).lower()
		part1 = str(dataCleaned.text.loc[baseRow])
		part2 = str(dataCleaned.label.loc[row])
		dataCleaned.text.loc[baseRow] = part1 + ' ' + part2
	
	print 'Merging split newline cells...'
	dataCleaned.drop(weirdRows, inplace = True)
	emptyRows = dataCleaned[pd.isnull(dataCleaned.label)].index
	dataCleaned.drop(emptyRows, inplace = True)
	
	print 'Dealing with suspicious duplicates...'
	textCounts = dataCleaned.text.value_counts(dropna = False)
	doubles = textCounts[textCounts.values == 2]
	
	duplicateRowsToDrop = []
	for post in list(doubles.index):
		if len(str(post)) < 20:
			continue
		doubleIndex =  list((dataCleaned[dataCleaned.text == post]).index)
		duplicateRowsToDrop.append(doubleIndex[1])
	
	print 'Copying and writing to file...'
	dataTrimmed = dataCleaned.drop(duplicateRowsToDrop)
	dataTrimmed.to_csv(outputPath, sep = '\t', header = False, index = False)

if __name__ == '__main__':
	script, inputPath, outputPath = sys.argv
	cleanData(inputPath, outputPath)
	print 'Wrote clean data to' + outputPath
