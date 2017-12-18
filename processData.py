import numpy as np
import sys
import pandas as pd
import time



def getData(testData=False):
	if(testData):
		df = pd.read_csv('./data/mnist_test.csv', header=None, engine='c', na_filter=False)
		return df.values

	#create pandas dataframe from .csv file
	df = pd.read_csv('./data/mnist_train.csv', header=None, engine='c', na_filter=False)	

	return df.values


if __name__ == '__main__':
	getData()