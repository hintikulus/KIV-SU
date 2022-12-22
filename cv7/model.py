import numpy as np
import utils


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

class Perceptron:

    def __init__(self, x_dim, num_classes, hidden_size, reg=0.0):
        self.theta = np.zeros((x_dim + 1) * hidden_size + (hidden_size + 1) * num_classes)
        self.init_weights()
        self.cost_history = []
        self.theta_history = []
        self.reg = reg
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.theta1_shape = [(x_dim + 1), hidden_size]
        self.theta2_shape = [(hidden_size + 1), num_classes]
        self.theta1_size = np.prod(self.theta1_shape)
        self.reg_mask = np.concatenate([np.zeros(hidden_size), np.ones(x_dim * hidden_size), np.zeros(num_classes), np.ones(hidden_size * num_classes)])

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        # TODO
        return None

    def get_positive_scores(self, X):
        """
        Computes the probability of classification to the positive class
        :param X: Input data
        :return:
        """
        # TODO
        return None

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        # TODO
        return None

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        # TODO
        return None

    def update(self, theta, cost):
        # print("%s : grad = %s, cost = %s" % (str(self.theta), str(G), str(self.__cost)))
        self.theta = theta
        self.theta_history.append(np.copy(self.theta))
        self.cost_history.append(cost)

    def init_weights(self):
        # TODO
        pass
