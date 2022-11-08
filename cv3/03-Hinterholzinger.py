import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from scipy.stats import linregress


class LinearRegression:
    def __init__(self):
        self.theta = np.array([0., 0.])

        self.cost_history = None
        self.theta_history = None

    def predict(self, X):
        """
        Computes the prediction (hzpothesis) of the linear regression
        :param X: input data as row vectors
        :return: vector of predicted outputs
        """

        return np.sum(np.multiply(np.reshape(X, (-1, 2)), self.theta), axis=1)

    def cost(self, X, y):
        """
        Computes the loss function of a linear regression (mean square error)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return: Loss value
        """

        J_part = (self.predict(X) - y) ** 2

        J = 0.5 * np.sum(J_part) / np.size(y)

        return J

    def train(self, X, y, lr, k, epsilon=None):
        """
        Trains the linear regression model (finds optimal parameters)
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :param lr: learning rate
        :param k: number of steps
        :param epsilon:
        :return:
        """

        n = np.size(X, axis=0)
        nr = 1/n

        q = nr * lr

        temp = np.zeros(np.size(self.theta))
        self.theta_history = []
        self.cost_history = []

        i = 0
        while i < k:
            hxy = self.predict(X) - y

            sum = np.array([
                np.sum(hxy),
                np.sum(hxy * X[:, 1])
            ])

            self.theta -= q * sum

            self.theta_history.append(self.theta.copy())
            self.cost_history.append(self.cost(X, y).copy())

            i += 1

        self.theta_history = np.array(self.theta_history)
        self.cost_history = np.array(self.cost_history)

        return self

def load_data(fn):
    """
    Loads the data for the assignment
    :param fn: file path
    :return: tuple: (input coordinates as the matrix of row vectors of the data, where each vector is in the form: [1, x],
                    expected outputs)
    """
    x_ = []
    y_ = []

    with open(fn, 'r') as f:
        for line in f:
            x, y = line.split(",")
            x_.append(float(x))
            y_.append(float(y))

    return np.stack([np.ones(len(x_)), np.array(x_)], axis=1), np.array(y_)


def plot_data(X, y):
    """
    Plots the data into a coordinates system.
    :param X:
    :param y:
    :return:
    """
    plt.figure()
    plt.scatter(X[:, 1], y, marker="x", color='red')
    plt.xlabel("City population (×1e5)")
    plt.ylabel("Profit (×1e5 $)")
    plt.savefig("data.pdf")
    plt.show()


def plot_regression(model, X, y):
    """
    Plots the data with the regression line.
    :param model: Linear regression model
    :param X: inputs
    :param y: expected outputs
    :return:
    """
    plt.figure()
    plt.scatter(X[:, 1], y, marker="x", color='red')
    x1 = np.min(X, axis=0)
    x2 = np.max(X, axis=0)
    y1 = model.predict(x1)
    y2 = model.predict(x2)
    plt.plot([x1[1], x2[1]], [y1, y2], color='blue')
    plt.legend(["Linear regression", "Training data"])
    plt.xlabel("City population (×1e5)")
    plt.ylabel("Profit (×1e5 $)")
    plt.savefig("regression.pdf")
    plt.show()


def plot_cost(model, X, Y):
    """
    Plots the loss value according to changing theta parameters
    :param model: Linear regression model
    :param X: input data as row vectors
    :param y: vector of the expected outputs
    :return:
    """
    dummy_model = LinearRegression()
    a1 = -2
    a2 = 4
    b1 = -30
    b2 = 30

    a_space = np.linspace(a1 - 1, a2 + 1)
    b_space = np.linspace(b1 - 1, b2 + 1)

    A, B = np.meshgrid(a_space, b_space)
    Z = np.zeros_like(A)
    for i, a in enumerate(a_space):
        for j, b in enumerate(b_space):
            dummy_model.theta = np.array([b, a])
            Z[j, i] = dummy_model.cost(X, Y)
    plt.figure()
    plt.contour(A, B, Z, levels=30)
    plt.xlim((a1 - 1, a2 + 1))
    plt.ylim((b1 - 1, b2 + 1))
    # TODO plot linear regression progress
    plt.scatter(model.theta_history[:, 1], model.theta_history[:, 0], marker=".", color="r", s=1)
    plt.scatter(model.theta[1], model.theta[0], marker="x", color="b")

    plt.savefig("cost.pdf")
    plt.show()


def plot_surf(X, y):
    """
    3d plot of the surface of the loss with regard to parameters theta
    :param X: input data as row vectors
    :param y: vector of the expected outputs
    :return:
    """
    model = LinearRegression()
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-10, 10, 100)
    Theta1, Theta0 = np.meshgrid(theta1_vals, theta0_vals)
    J_vals = np.zeros_like(Theta1)
    for i in range(len(theta1_vals)):
        for j in range(len(theta0_vals)):
            model.theta = [theta0_vals[j], theta1_vals[i]]
            J_vals[i][j] = model.cost(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(Theta1, Theta0, J_vals, cmap=cm.coolwarm)
    plt.savefig("convergence3d.pdf")
    plt.show()

def plot_alphas(X_, y_, alphas, steps, normal):

    legend = []
    plt.figure()

    for a in alphas:
        model = LinearRegression().train(X_, y_, a, steps)
        plt.plot(np.linspace(1, steps, num=steps), model.cost_history)
        legend.append("alpha = " + str(a))

    x = np.linspace(1, 1000, num=1000)

    #plt.plot(x, normal[0] * x + normal[1], '-', color='green')
    #legend.append("Normal")

    linreg = LinearRegression()
    linreg.theta = normal

    plt.plot(x, lin_reg.cost(X_, y_) + 0*x, '--', color='green')
    legend.append("Analytical Solution")

    plt.legend(legend)

    plt.axis([-2, steps, 4.2, 6])
    plt.xlabel("Step")
    plt.ylabel("J(theta)")
    plt.savefig("alpha.pdf")
    plt.show()

def f(x):
    return

if __name__ == '__main__':
    steps = 1000
    X_, y_ = load_data("data.txt")

    plot_data(X_, y_)

    lin_reg = LinearRegression()
    alpha = 0.024
    lin_reg.train(X_, y_, alpha, steps)
    print("Theta found by gradient descent: " + str(lin_reg.theta))

    predict1 = lin_reg.predict(np.array([1, 3.5]))
    print(f'For population = 35,000, we predict a profit of {predict1 * 10000}')
    predict2 = lin_reg.predict(np.array([1, 7]))
    print(f'For population = 70,000, we predict a profit of {predict2 * 10000}')

    plot_regression(lin_reg, X_, y_)
    plot_surf(X_, y_)
    plot_cost(lin_reg, X_, y_)
    # TODO plot convergence graphs for different values of alpha along with an analytical solution using normal equation

    # Pokus o výpočet parametrů pomocí normální rovnice
    mXt = X_.transpose()
    mX = X_

    A = np.matmul(mXt, mX)
    A = np.linalg.inv(A)
    A = np.matmul(A, mXt)
    A = np.matmul(A, y_.transpose())

    print("Theta nalezena normalni rovnici:", A)

    alphas = [0.005, 0.015, 0.020, 0.024, 0.0241, 0.02425, 0.0243]
    plot_alphas(X_, y_, alphas, 1000, A)
