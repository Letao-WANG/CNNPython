import numpy as np

LEARNING_RATE = 0.1
EPOCHS = 1000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_predict):
    return ((y_true - y_predict) ** 2).mean()
