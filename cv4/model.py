import numpy as np


class LinearRegression:

    def __init__(self, x_dim):
        self.theta = np.zeros(x_dim)
        self.cost_history = []
        self.theta_history = []

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """
        # TODO
        return self.theta.dot(X.transpose())

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """
        # TODO
        predictions = self.predict(X)
        cost = (1 / (2 * y.shape[0])) * np.sum(np.square(predictions - y))
        return cost

    def grad(self, X, y):
        """
        Computes the gradient of the loss function with regard to the parameters theta
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Gradient
        """
        # TODO
        return np.matmul(X.transpose(), (self.predict(X) - y))

    def analytical_solution(self, X, y):
        """
        Computes analytical solution of the least-squares method (normal equation)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        # TODO
        Xt = X.transpose()
        part = np.linalg.inv(Xt.dot(X))

        self.theta = part.dot(Xt).dot(y)
        pass
