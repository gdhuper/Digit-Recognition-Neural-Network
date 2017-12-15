import sys
import math
import numpy as np
from numba import jit
import time
from processData import getData
import pandas as pd


class NeuralNetwork():

	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		self.inputNodes = inputNodes
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes

		#learning rate 
		self.learningRate = learningRate

		#seed for consistent values
		#np.random.seed(1)

		#weights (synapses) between input layer nodes and hidden layer nodes
		self.syn0 = (2*np.random.random((self.inputNodes, self.hiddenNodes)) - 1).T

		#weights (synapses) between hidden layer nodes and output layer nodes
		self.syn1 = (2*np.random.random((self.hiddenNodes, self.outputNodes)) - 1).T


		self.normalize_vals = lambda x: (x / 255.0 * 0.99) + 0.01


	def train(self, inputValues, outputValues):
		"""
		Trains neural network (feed forward) and backpropogation
    	:param expected: input node values (vectors from .csv file) and output node values based on given label
    	:return: None
    	"""
    	#### forward pass ###
    	
    	#dot product between input layer and hidden layer
		x_hidden = self.dotproduct(self.syn0, self.normalize_vals(inputValues))

		# calculating sigmoid value for hidden layer nodes
		o_hidden = self.sigmoid(x_hidden)

		# dot product between hidden layer and output
		x_output_layer = self.dotproduct(self.syn1, o_hidden)

		# calculating sigmoid for output layer
		o_output_layer = self.sigmoid(x_output_layer).T

		# calculating error rate for final output
		delta = outputValues - o_output_layer

		print("delta\n", delta)

		### backpropogation ###



	def query(self):
		pass

	def sigmoid(self, x, deriv=False):
		if deriv == True:
			return x * (1.0 - x)
		return 1.0 / (1.0 + np.exp(-x))

	def dotproduct(self, weights, inputs):
		#dot product using numpy
		return np.dot(weights, inputs) 



#helper function to unit test function
def test():
	nn = NeuralNetwork("inputNodes", "hiddleNodes", "outNodes", "learningRate")

	a = np.random.randint(255, size=(100,784))
	b = np.random.randint(255, size=(784,1))

	start_time = time.time()
	print(nn.dotproduct(a, b))
	t1 = (time.time() - start_time)
	print("--- %s seconds ---" % t1)

	start_time2 = time.time()
	print(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
	dotProduct(a, b)
	t2 = (time.time() - start_time2)
	print("--- %s seconds ---" % t2)

	print("Pure Python: ", t1)
	print("OpenCL: ", t2)
	print("Pure Python is faster" if t1 < t2 else "OpenCl is faster")
	



@jit
def main():
	inputNodes = 784
	hiddleNodes = 100
	outNodes = 10

	learningRate = 0.5

	#create instance of neural net with input parameters
	nn = NeuralNetwork(inputNodes, hiddleNodes, outNodes, learningRate)

	inputVectors = getData()

	epoch = 1

	for i in range(epoch):
		for vector in inputVectors:
			#strip label from each vector
			label = vector[0]

			#values for input nodes in input layer
			inodes = vector[1:]

			#reshaping 1d array to 2D array for dot product compatibility 
			inNodes = np.reshape(inodes, (inputNodes, 1))

			#expected output values for output layer
			outputValues = np.zeros(outNodes) + 0.01

			#set result node to 0.99
			outputValues[label] = 0.99

			outNodes = np.reshape(outputValues, (10, 1))

			nn.train(inNodes, outputValues)
			break





	
	


if __name__ == '__main__':
	main()



# # train the network
# epochs = 5

# for e in range(epochs):
#     for record in training_data_list:
#         all_values = record.split(',')
#         inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#         targets = numpy.zeros(output_nodes) + 0.01
#         # all_values[0] is the target label for this record
#         targets[int(all_values[0])] = 0.99
#         n.train(inputs, targets)
#         pass
#     pass