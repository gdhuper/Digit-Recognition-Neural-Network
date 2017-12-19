import sys
import math
import numpy as np
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
		np.random.seed(1)

		#weights (synapses) between input layer nodes and hidden layer nodes
		self.syn0 = 2*np.random.random((self.hiddenNodes, self.inputNodes)) - 1

		#weights (synapses) between hidden layer nodes and output layer nodes
		self.syn1 = 2*np.random.random((self.outputNodes ,self.hiddenNodes)) - 1
		

		#lambda function to normalize vector values
		self.normalize_vals = lambda x: (np.asfarray(x) / 255.0 * 0.99) + 0.01


	def train(self, inputValues, outputValues):
		"""
		Trains neural network (feed forward) and backpropogation
    	:param expected: input node values (vectors from .csv file) and output node values based on given label
    	:return: None
    	"""
    	### forward pass ###
        
    	#dot product between input layer and hidden layer
		x_hidden = self.dotproduct(self.syn0, inputValues)

		

		# calculating sigmoid value for hidden layer nodes
		o_hidden = self.sigmoid(x_hidden)

		# dot product between hidden layer and output layer
		x_output_layer = self.dotproduct(self.syn1, o_hidden)

		# calculating sigmoid for output layer
		o_output_layer = self.sigmoid(x_output_layer)


		# calculating error rate for final output
		final_error = outputValues - o_output_layer

		#print("Error: " + str(np.mean(np.abs(final_error))))
		
		### backpropogation ###

		#calculating error for hidden layer
		hidden_layer_error = self.dotproduct(self.syn1.T, final_error)
		

		#updating weights between hidden layer and output layer using gradient descent
		t_layer1 = final_error * (o_output_layer * (1.0 - o_output_layer))
		self.syn1 += self.learningRate * np.dot(t_layer1, o_hidden.T)

		#updating weights between input layer and hidden layer using gradient descent
		t_layer0 = hidden_layer_error * (o_hidden * (1.0 - o_hidden))
		self.syn0 += self.learningRate * np.dot(t_layer0, inputValues.T)
		



	def query(self, inputValues):
		#dot product between input layer and hidden layer
		x_hidden = self.dotproduct(self.syn0, inputValues)

		# calculating sigmoid value for hidden layer nodes
		o_hidden = self.sigmoid(x_hidden)

		# dot product between hidden layer and output
		x_output_layer = self.dotproduct(self.syn1, o_hidden)

		# calculating sigmoid for output layer
		o_output_layer = self.sigmoid(x_output_layer)

		return o_output_layer


	def sigmoid(self, x, deriv=False):
		if deriv == True:
			return x * (1.0 - x)
		return 1.0 / (1.0 + np.exp(-x))

	def dotproduct(self, weights, inputs):
		#dot product using numpy
		return np.dot(weights, inputs) 


	

def main():
	inputNodes = 784
	hiddleNodes = 100
	outNodes = 10

	learningRate = 0.3

	#create instance of neural net with input parameters
	nn = NeuralNetwork(inputNodes, hiddleNodes, outNodes, learningRate)

	inputVectors = getData()

	#training neural net 
	epoch = 3

	start_time = time.time()
	
	
	for i in range(epoch):
		for vector in inputVectors:

			#strip label from each vector
			label = vector[0]

			#values for input nodes in input layer
			inodes = nn.normalize_vals(vector[1:]).reshape(inputNodes, 1)
		
			#expected output values for output layer
			outputValues = np.zeros(outNodes) + 0.01

			#set result node to 0.99
			outputValues[label] = 0.99
			
			outNodeValues = np.reshape(outputValues, (10, 1))

			nn.train(inodes, outNodeValues)		
			
			
	t1 = (time.time() - start_time)
	
	print("--- %s seconds ---" % t1)

	correct_predictions = 0

	#testing neural net and calculating accuracy 
	testVectors = getData(testData=True)

	for vector in testVectors:
		#actual label
		true_label = vector[0]

		inodes = nn.normalize_vals(vector[1:]).reshape(inputNodes, 1)

		predicted_outputs = nn.query(inodes)

		#predicted label
		predicted_label = np.argmax(predicted_outputs)

		print("True label : {}, Predicated Label: {}".format(true_label, predicted_label))

		if predicted_label == true_label:
			correct_predictions += 1

	print("accuracy : " + str(correct_predictions / len(testVectors)))
	#accuracy = 93-94% with epoch = 3, and learning rate = 0.3


	

if __name__ == '__main__':
	main()

