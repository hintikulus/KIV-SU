import matplotlib.pyplot as plt
import numpy as np
from model import LinearRegression
from itertools import combinations, product
import matplotlib.colors as mcolors
import utils


def plot_regression(X, mu, sigma, y, lin_reg):
    plt.figure()
    for i in set(X[:, 2]):
        x1 = np.min(X[:, 1])
        x1 = np.array([x1, i])
        x2 = np.max(X[:, 1])
        x2 = np.array([x2, i])
        y1 = lin_reg.predict(utils.add_one((x1 - mu) / sigma))
        y2 = lin_reg.predict(utils.add_one((x2 - mu) / sigma))

        plt.scatter(X[X[:, 2] == i, 1], y[X[:, 2] == i], marker="x", color=list(mcolors.BASE_COLORS.values())[int(i)])
        plt.plot([x1[0], x2[0]], [y1, y2], color=list(mcolors.BASE_COLORS.values())[int(i)])
        plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.xlabel("Size in square feet")
    plt.ylabel("Prize")
    plt.legend([str(j) + " rooms" for j in set(X.astype(int)[:, 2])])
    plt.savefig("regression.pdf")
    plt.show()


def plot_data(X, y):
    plt.figure()
    for i in set(X[:, 2]):
        plt.scatter(X[X[:, 2] == i, 1], y[X[:, 2] == i], marker="x", color=list(mcolors.BASE_COLORS.values())[int(i)])
    plt.xlabel("Size in square feet")
    plt.ylabel("Prize")
    plt.legend([str(j) + " rooms" for j in set(X.astype(int)[:, 2])])
    plt.savefig("data.pdf")
    plt.show()


def __plot_cost_partial(X, y, dim1, dim2, theta_history, dummy_model, true_theta, model, alpha):
    print("plotting cost[theta%d, theta%d]" % (dim1, dim2))
    a_history = theta_history[:, dim1]
    b_history = theta_history[:, dim2]
    for dim in set(range(len(model.theta))) - {dim1, dim2}:
        dummy_model.theta[dim] = model.theta[dim]

    a1, a2 = cost_bounds(a_history, true_theta[dim1])
    b1, b2 = cost_bounds(b_history, true_theta[dim2])

    da = (a2 - a1)
    db = (b2 - b1)

    if da > db:
        D = (da - db) / 2
        b2 += D
        b1 -= D
    else:
        D = (db - da) / 2
        a2 += D
        a1 -= D

    a_space = np.linspace(a1, a2, 256)
    b_space = np.linspace(b1, b2, 256)

    A, B = np.meshgrid(a_space, b_space)
    Z = np.zeros_like(A)
    for i, a in enumerate(a_space):
        for j, b in enumerate(b_space):
            dummy_model.theta[dim1] = a
            dummy_model.theta[dim2] = b
            Z[j, i] = dummy_model.cost(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(a1, a2)
    plt.ylim(b1, b2)
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    ax.set_aspect('equal', adjustable='box')
    plt.contour(A, B, Z, levels=16)
    plt.plot(a_history[:-1], b_history[:-1], color='red')
    plt.scatter(a_history[-1], b_history[-1], marker="x", color='red', s=128)
    # plt.scatter(true_theta[dim1], true_theta[dim2], marker="+", color='green', s=128)
    plt.title("Alpha = %.6f, points = %d" % (alpha, len(theta_history)))
    plt.xlabel("Theta %d" % dim1)
    plt.ylabel("Theta %d" % dim2)

    plt.savefig("cost_%d_%d.pdf" % (dim1, dim2))
    # plt.clf()


def cost_bounds(history, true_theta):
    v1 = np.min([cost for cost in history if np.isfinite(cost)])
    v2 = np.max([cost for cost in history if np.isfinite(cost)])
    if np.abs(true_theta - v1) > np.abs(true_theta - v2):
        v2 = 2 * true_theta - v1
    else:
        v1 = 2 * true_theta - v2
    return v1, v2


def plot_cost(X, y, model, alpha):
    plt.figure()
    dummy_model = LinearRegression(X.shape[1])
    theta_history = np.array(model.theta_history)

    for d1, d2 in combinations(np.arange(len(model.theta)), 2):
        __plot_cost_partial(X, y, d1, d2, theta_history, dummy_model, model.theta, model, alpha)
    plt.show()


def plot_convergence(X, y, model, alpha):
    plt.figure()
    if alpha is None:
        print("You must first train the model.")
        exit(1)

    print("plotting convergence")

    theta_history = np.array(model.theta_history)

    analytic = LinearRegression(X.shape[1])
    analytic.analytical_solution(X, y)

    Xs = np.linspace(0, len(theta_history), len(theta_history))

    for i in range(len(model.theta)):
        plt.plot(Xs, theta_history[:, i])
        plt.plot(Xs, np.repeat(analytic.theta[i], len(Xs)), "--")

    lgd = plt.legend(
        [b + a for a, b in
         product([str(i) for i in range(len(model.theta))], ["Numeric theta ", "Analytic theta "])],
        title="Legend",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.title("Alpha = %.6f" % alpha)

    plt.xlabel("Step")
    plt.ylabel("Theta value")
    plt.savefig("convergence_theta.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()
    plt.figure()

    plt.plot(Xs, model.cost_history)

    lgd = plt.legend(
        ["Cost"],
        title="Legend",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.title("Alpha = %.6f" % alpha)

    plt.xlabel("Step")
    plt.ylabel("Cost value")
    plt.savefig("convergence_cost.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()
