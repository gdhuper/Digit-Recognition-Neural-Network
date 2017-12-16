import numpy as np
import sys
from numba import jit
import pandas as pd
import time


@jit
def getData(testData=False):
	if(testData):
		df = pd.read_csv('./data/mnist_test.csv', header=None, engine='c', na_filter=False)
		return df.values


	#start_time1 = time.time()
	df = pd.read_csv('./data/mnist_train.csv', header=None, engine='c', na_filter=False)	
	#t2 = (time.time() - start_time1)
	#print("--- %s seconds ---" % t2)

	return df.values


if __name__ == '__main__':
	getData()