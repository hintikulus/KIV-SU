import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


def plot_decision_boundary(X, y, model):
    """
    Plots the data and decision boundary.
    :param X: input data
    :param y: gold labels
    :param model: logistic regression model
    :return:
    """
    plt.figure()
    # TODO plot the decision boundary line


    plt.scatter(X[y == 0, 1], X[y == 0, 2], marker="x")
    plt.scatter(X[y == 1, 1], X[y == 1, 2], marker="o")
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.legend(["Decision boundary", "Not admitted", "Admitted"])
    plt.savefig("DB.pdf")
    plt.show()


def plot_data(X, y):
    plt.figure()
    plt.scatter(X[y == 0, 1], X[y == 0, 2], marker="x")
    plt.scatter(X[y == 1, 1], X[y == 1, 2], marker="o")
    plt.ticklabel_format(style='sci', scilimits=(0, 0))
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.legend(["Not admitted", "Admitted"])
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


def cost_bounds(history, true_theta):
    v1 = np.min([cost for cost in history if np.isfinite(cost)])
    v2 = np.max([cost for cost in history if np.isfinite(cost)])
    if np.abs(true_theta - v1) > np.abs(true_theta - v2):
        v2 = 2 * true_theta - v1
    else:
        v1 = 2 * true_theta - v2
    return v1, v2


def plot_cost(X, y, model, alpha, dummy_model):
    theta_history = np.array(model.theta_history)

    for d1, d2 in combinations(np.arange(len(model.theta)), 2):
        __plot_cost_partial(X, y, d1, d2, theta_history, dummy_model, model.theta, model, alpha)
    plt.show()


def plot_convergence(model, alpha):
    plt.figure()
    if alpha is None:
        print("You must first train the model.")
        exit(1)

    print("plotting convergence")

    theta_history = np.array(model.theta_history)
    Xs = np.linspace(0, len(theta_history), len(theta_history))
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
