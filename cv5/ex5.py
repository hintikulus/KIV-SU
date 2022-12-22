import numpy as np
from model import LogisticRegression
import utils
import visualize as vis
from optimize import GradientDescent


def load_points(fn):
    x_ = []
    y_ = []

    with open(fn, 'r') as f:
        for line in f:
            vals = line.split(",")
            _x = vals[:-1]
            _y = vals[-1]
            x_.append([float(x) for x in _x])
            y_.append(float(_y))

    return np.array(x_), np.array(y_)

if __name__ == '__main__':
    X, y = load_points("data1.txt")
    X_norm, mu, sigma = utils.normalize_features(X)
    X = utils.add_one(X)
    X_norm = utils.add_one(X_norm)
    log_reg = LogisticRegression(len(X[0]))
    print("Cost at ititial theta (zeros): %f" % log_reg.cost(X_norm, y))
    m, n = X.shape
    vis.plot_data(X, y)
    opt = GradientDescent(log_reg, alpha=25, num_iters=1000)
    opt.optimize_full_batch(X_norm, y)
    print("Cost at theta: %f" % log_reg.cost(X_norm, y))
    vis.plot_decision_boundary(X_norm, y, log_reg)
    vis.plot_cost(X_norm, y, log_reg, opt.alpha, LogisticRegression(len(X[0])))
    vis.plot_convergence(log_reg, opt.alpha)
    p = log_reg.predict(X_norm)
    print("Train accuracy: %d%%" % (100 * np.sum(y == p) / len(y)))
    perm = np.random.permutation(len(y))
    X_perm = X_norm[perm, :]
    y_perm = y[perm]
    y_pred_test = utils.cross_validation(X_perm, y_perm, 5, lambda model: GradientDescent(model), lambda: LogisticRegression(X.shape[1]))
    print("Test accuracy: %d%%" % (100 * np.sum(y_perm == y_pred_test) / len(y)))
    # TODO
    x = [[45, 85]]
    x = np.array(x)
    x = utils.add_one((x - mu) / sigma)

    for xi in x:
        pred = log_reg.get_positive_score(xi)
    print("A student with scores 45, 85, will be admitted with probability of %f%%" % (pred * 100))
