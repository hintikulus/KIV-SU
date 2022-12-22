import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from scipy.stats import linregress


class LinearRegression:

    def __init__(self, x_dim):
        self.theta = np.zeros(x_dim)
        self.cost_history = []
        self.theta_history = []
        self.reg = 0

    def hyp(self, X):
        return np.sum(self.predict(X))

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        # TODO

        return self.theta.dot(X.T)

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        # TODO
        predictions = self.predict(X)
        cost = (1 / (2 * y.shape[0])) * (np.sum(np.square(predictions - y)) + self.reg * np.sum(np.square(self.theta)))
        return cost

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        # TODO
        return np.matmul(X.T, (self.predict(X) - y))

    def analytical_solution(self, X, y):
        """
        Computes analytical solution of the least-squares method (normal equation)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        # TODO
        size = X.shape[1]
        Xt = X.transpose()

        XtX = Xt.dot(X)

        ones = np.eye(size)
        ones[0][0] = 0

        reg = self.reg * ones

        XtX = XtX + reg

        part = np.linalg.pinv(XtX)

        self.theta = part.dot(Xt).dot(y)
        pass



class LogisticRegression:

    def __init__(self, x_dim):
        self.theta = np.zeros(x_dim)
        self.cost_history = []
        self.theta_history = []
        self.reg = 0

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        # TODO
        ret = []
        for p in X:
            ret.append(self.get_positive_score(p) >= 0.5 if 1 else 0)
        return ret

    def get_positive_score(self, X):
        """
        Computes the probability of classification to the positive class
        :param X: Input data
        :return:
        """
        # TODO
        exp = np.exp(np.dot(X, self.theta) * (-1))

        return 1 / (1 + exp)

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        # TODO
        m = X.shape[0]

        sig = self.get_positive_score(X)
        reg = (self.reg / (2 * m)) * (np.sum(self.theta ** 2))

        log = y * np.log(sig) + (1 - y) * np.log(1 - sig)
        sum_log = np.sum(log)
        total_cost = (-1 / m) * sum_log + reg

        return total_cost

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        # TODO

        m = X.shape[0]
        sig = self.get_positive_score(X)
        return (1 / m) * np.dot(X.T, sig - y)

    def update(self, theta, cost):
        # print("%s : grad = %s, cost = %s" % (str(self.theta), str(G), str(self.__cost)))
        self.theta = theta
        self.theta_history.append(np.copy(self.theta))
        self.cost_history.append(cost)
