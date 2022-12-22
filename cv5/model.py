import numpy as np


class LogisticRegression:

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
        # e na X*theta
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

        log = y * np.log(sig) + (1 - y) * np.log(1 - sig)
        sum_log = np.sum(log)
        total_cost = (-1 / m) * sum_log

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


class OneVsAll:

    def __init__(self, model_gen, opt_gen):
        """
        One-vs-all technique implementation
        :param model_gen: a generator function which creates a new model (with number of input features as a parameter)
        :param opt_gen: a generator function which creates a new optimizer (with model as a parameter)
        """
        self.model_gen = model_gen
        self.opt_gen = opt_gen
        self.models = []

    def predict(self, X):
        """
        Predicts the class for each datapoint (row of X)
        :param X: input data
        :return:
        """
        predicts = []
        print("Prediction...")
        for model in self.models:
            predicts.append(model.get_positive_score(X))
        predicts = np.array(predicts)
        result = np.argmax(predicts, axis=0)
        print("Prediction finished")
        return result

    def train(self, X, y):
        """
        Trains one-vs-all classifier (separate logistic regression for each class)
        :param X: input data
        :param y: gold classes
        :return:
        """
        # TODO

        for group in range(10):
            ys = []

            for yi in y:
                ys.append(group == yi if 1 else 0)

            print("Group %d" % group)
            model = self.model_gen(X.shape[1])
            self.models.append(model)
            op = self.opt_gen(model)

            op.optimize_full_batch(X, np.array(ys))
        pass

