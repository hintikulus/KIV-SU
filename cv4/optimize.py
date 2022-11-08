import numpy as np
import math


class Optimizer:

    def __init__(self, model):
        self.model = model
        self.iter = 0

    def step(self, X, y):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        raise NotImplementedError("Method not yet implemented.")

    def converged(self):
        """

        :return: True if the gradient descent iteration ended
        """
        raise NotImplementedError("Method not yet implemented.")

    def optimize_full_batch(self, X, y):
        """
        Runs the optimization processing all the data at each step
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        while not self.converged():
            self.step(X, y)
            self.iter += 1


class GradientDescent(Optimizer):

    def __init__(self, model, alpha=0.0005, num_iters=1000, min_cost=0, min_theta_diff=0, **options):
        super(GradientDescent, self).__init__(model)
        self.options = options
        self.alpha = alpha
        self.num_iters = num_iters
        self.min_cost = min_cost
        self.min_theta_diff = min_theta_diff
        self.cost = np.Inf

    def step(self, X, y):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        new_theta = self.model.theta - self.alpha * (1 / y.shape[0]) * self.model.grad(X, y)
        self.model.theta_history.append(self.model.theta)
        self.model.cost_history.append(self.model.cost(X, y))

        self.model.theta = new_theta
        self.cost = self.model.cost_history[-1]
        return None

    def converged(self):
        """
        :return: True if the gradient descent iteration ended
        """
        # TODO

        theta_diff = np.linalg.norm(self.model.theta_history[-1] - self.model.theta) \
            if len(self.model.theta_history) > 1 \
            else math.inf
        # if J(i) > J(i-1)
        #return self.iter == self.num_iters or self.cost <= self.min_cost or theta_diff <= self.min_theta_diff

        # Pocet iteraci
        if(self.iter >= self.num_iters): return True

        # Rozdil hodnot
        if len(self.model.theta_history) > 1:
            diff = np.linalg.norm(self.model.theta_history[-1] - self.model.theta)
            diff = diff if diff > 0 else -diff

            if diff <= self.min_theta_diff: return True

        # Minimalni cena
        if self.cost <= self.min_cost: return True

        # Jinak nepravda
        return False


