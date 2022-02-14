from util import *
import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        """
        :param weights: e.g. [1..300]
        :param bias: e.g. [1]
        """
        self.weights = weights
        self.bias = bias
        self.h = 0

    def __repr__(self):
        return "weights: " + str(self.weights) + ", bias: " + str(self.bias)

    def get_total(self, inputs):
        """

        :param inputs: [784, 1]
        :return: scalar
        """
        return np.dot(self.weights, inputs) + self.bias

    def feedforward(self, inputs):
        """

        :param inputs:
        :return: scalar
        """
        total = self.get_total(inputs)
        self.h = sigmoid(total)
        return sigmoid(total)

    def update(self, inputs, y_true):
        """
        :param inputs: vector [784]
        :param y_true: scalar
        :return:
        """
        g = self.get_total(inputs)  # g is scalar
        y_predict = self.feedforward(inputs)  # y_predict is scalar
        self.h = y_predict
        # L = (y_true - y_predict)**2
        d_l_d_y_predict = 2 * (y_predict - y_true)
        d_y_predict_d_g = derivative_sigmoid(g)
        d_g_d_w = inputs
        d_g_d_b = 1

        d_l_d_w = d_l_d_y_predict * d_y_predict_d_g * d_g_d_w
        d_l_d_b = d_l_d_y_predict * d_y_predict_d_g * d_g_d_b

        self.weights -= LEARNING_RATE * d_l_d_w
        self.bias -= LEARNING_RATE * d_l_d_b


class NeuralNetwork:
    def __init__(self, sizes):
        """
        For example, if the list was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.

        For mnist, sizes = [784, 300, 300, 1]
        :param sizes: list of number neurons
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # list of neurons
        self.layers = []

        self.initialize()

    def __repr__(self):
        return "The shape of this nn: " + str(self.sizes)

    def initialize(self):
        for num_layer in range(self.num_layers):
            neurons = []
            for num_neuron in range(self.sizes[num_layer]):
                # layer 0 is input data
                if num_layer is not 0:
                    # weights = [number of prev layer, 1], bias = [1]
                    neurons.append(Neuron(np.random.randn(self.sizes[num_layer - 1]), np.random.randn()))
            self.layers.append(neurons)

    def train(self, inputs, result):
        """

        :param inputs: array of 28 * 28
        :param result: scalar
        :return:
        """
        data = inputs.ravel()
        new_data = []
        for neurons in self.layers[0]:
            neurons.update(data, result)
            new_data.append(neurons.h)

        # print(new_data)

        # data2 = []
        # for neurons in self.layers[1]:
        #     neurons.update(new_data, result)
        #     data2.append(neurons.h)

        # for neurons in self.layers[2]:
        #     neurons.update(data2, result)

    def result(self):
        neuron = self.layers[len(self.layers)-1][0]
        return neuron.h
