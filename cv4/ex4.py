import numpy as np
from model import LinearRegression
import visualize as vis
import utils as utils
from optimize import GradientDescent


def load_data(path):
    x_ = []
    y_ = []

    with open(path, 'r') as f:
        for line in f:
            vals = line.split(",")
            _x = vals[:-1]
            _y = vals[-1]
            x_.append([float(x) for x in _x])
            y_.append(float(_y))

    return np.array(x_), np.array(y_)


if __name__ == '__main__':
    X, y = load_data("data1.txt")
    X_norm, mu, sigma = utils.normalize_features(X)
    X = utils.add_one(X)
    X_norm = utils.add_one(X_norm)
    linReg = LinearRegression(X.shape[1])
    m, n = X.shape
    vis.plot_data(X, y)
    opt = GradientDescent(linReg, alpha=0.15, num_iters=100)
    opt.optimize_full_batch(X_norm, y)
    vis.plot_regression(X, mu, sigma, y, linReg)
    vis.plot_cost(X_norm, y, linReg, opt.alpha)
    vis.plot_convergence(X_norm, y, linReg, opt.alpha)
    normal_eq = LinearRegression(X.shape[1])
    normal_eq.analytical_solution(X, y)
    x = [1650, 3]
    # TODO
    x = np.array(x)
    x2 = utils.add_one(x)
    x = utils.add_one((x - mu) / sigma)
    pred_normal = normal_eq.predict(x2)
    print("Predicted price of a 1650 sq-ft, 3 br house (using normal equation): %f" % pred_normal)
    pred_gd = linReg.predict(x)
    print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): %f" % pred_gd)
