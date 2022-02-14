from nn import *
from keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()
#
# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))


nn_sizes = [784, 300, 300, 1]
# nn_sizes = [7, 3, 3, 1]
nn = NeuralNetwork(nn_sizes)
# print([len(e) for e in nn.bias])
# print(nn.weights)
for _ in range(1000):
    nn.train(train_X[0], train_y[0])
    
print(nn.result())


