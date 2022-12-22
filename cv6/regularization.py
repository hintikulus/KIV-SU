#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

# custom imports
import utils
from model import LinearRegression, LogisticRegression
from optimize import GradientDescent
from visualize import plot_train_dev_curve


# ## Linear regression - regularized


def poly_features(X, p):
    n = X.shape[1]
    X_poly = np.zeros((X.shape[0], n * p))

    for i in range(1, p + 1):
        X_poly[:, (i - 1) * n:i * n] = X ** i
    return X_poly


def plot_fit(X, model, plt, mu, sigma):
    min_x = np.min(X)
    max_x = np.max(X)
    X_space = np.linspace(min_x - 15, max_x + 25)
    X_space_mapped = (poly_features(np.expand_dims(X_space, 1), 8) - mu) / sigma
    y = model.predict(utils.add_one(X_space_mapped))
    plt.plot(X_space, y)


def map_feature(X1, X2):
    degree = 7
    res = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i - j)) * (X2 ** j)))

    return res


def plot_decision_boundary(theta, axes):
    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    U, V = np.meshgrid(u, v)

    U = np.ravel(U)
    V = np.ravel(V)

    X_poly = map_feature(U, V)
    Z = X_poly @ theta

    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))

    cs = axes.contour(U, V, Z, levels=[0], cmap="rainbow")
    return cs


if __name__ == '__main__':

    data1 = loadmat('data1.mat')
    X_1 = data1["X"]
    y_1 = np.squeeze(data1["y"])

    # test data
    X1_test = data1["Xtest"]
    y1_test = np.squeeze(data1["ytest"])

    # validation data
    X1_val = data1["Xval"]
    y1_val = np.squeeze(data1["yval"])

    fig, ax = plt.subplots()
    ax.plot(X_1[:], y_1[:], 'rx')
    ax.set_xlabel('Change in water level (x)')
    ax.set_ylabel('Water flowing out of the dam (y)')
    plt.savefig("img/ex6-data1.pdf")
    fig.show()

    linReg = LinearRegression(X_1.shape[1] + 1)
    linReg_opt = GradientDescent(linReg, alpha=0.001)
    norm_eq = LinearRegression(X_1.shape[1] + 1)

    linReg.theta = np.array([1, 1])
    linReg.reg = 1
    J = linReg.cost(utils.add_one(X_1), y_1)
    grad = linReg.grad(utils.add_one(X_1), y_1)

    print(f'Cost at theta = [1 ; 1]: {J} \n(this value should be about 303.993192)\n')
    print(f'Gradient at theta = [1 ; 1]:  {grad} \n(this value should be about [-15.303016; 598.250744])\n')

    # Generates the train and cross validation set errors needed to plot a learning curve

    linReg.reg = 0
    err_tr_lr, err_val_lr = utils.learning_curve(X_1, y_1, X1_val, y1_val, linReg, linReg_opt.optimize_full_batch)

    plot_train_dev_curve(list(range(2, 13)), err_tr_lr, err_val_lr, "Learning Curve with Gradient Descent")

    err_tr_lr, err_val_lr = utils.learning_curve(X_1, y_1, X1_val, y1_val, norm_eq, norm_eq.analytical_solution)

    plot_train_dev_curve(list(range(2, 13)), err_tr_lr, err_val_lr, "Learning Curve with Normal Equation")

    # ## Polynomial linear regression - regularized

    norm_eq.analytical_solution(utils.add_one(X_1), y_1)
    print(norm_eq.theta)

    costNormal = norm_eq.cost(utils.add_one(X_1), y_1)
    print("Cost - NormalEQ", costNormal)

    # In[9]:

    X1_poly = poly_features(X_1, 8)
    X1_val_poly = poly_features(X1_val, 8)

    X1_poly_normalize, X1_poly_mu, X1_poly_sigma = utils.normalize_features(X1_poly)
    X1_val_poly_normalize, X1_val_poly_mu, X1_val_poly_sigma = utils.normalize_features(X1_val_poly)

    poly_model = LinearRegression(X1_poly_normalize.shape[1] + 1)
    poly_opt = GradientDescent(poly_model, alpha=0.001)
    poly_norm_eq = LinearRegression(X1_poly_normalize.shape[1] + 1)

    poly_model.reg = 0
    err_tr_lr, err_val_lr = utils.learning_curve(X1_poly_normalize, y_1, X1_val_poly_normalize, y1_val, poly_model,
                                                 poly_opt.optimize_full_batch)

    plot_train_dev_curve(list(range(2, 13)), err_tr_lr, err_val_lr, "Polynomial Learning Curve with Gradient Descent")

    err_tr_lr, err_val_lr = utils.learning_curve(X1_poly_normalize, y_1, X1_val_poly_normalize, y1_val, poly_norm_eq,
                                                 poly_norm_eq.analytical_solution)

    plot_train_dev_curve(list(range(2, 13)), err_tr_lr, err_val_lr, "Polynomial Learning Curve with Normal Equation")

    # validationCurve
    lambda_vec = np.array([0, 4, 10, 20])
    gradientOptions = {"iters": 400, "learningRate": 0.02, "regTerm": 0}

    errors_train, errors_val = utils.validation_curve(X1_poly_normalize, y_1, X1_val_poly_normalize, y1_val, poly_model,
                                                      poly_opt.optimize_full_batch, lambda_vec)

    plot_train_dev_curve(list(lambda_vec), errors_train, errors_val,
                         "Polynomial Validation Curve with Gradient Descent", xlabel="lambda value")

    errors_train, errors_val = utils.validation_curve(X1_poly_normalize, y_1, X1_val_poly_normalize, y1_val,
                                                      poly_norm_eq, poly_norm_eq.analytical_solution, lambda_vec)

    plot_train_dev_curve(list(lambda_vec), errors_train, errors_val, "Polynomial Validation Curve with Normal Equation",
                         xlabel="lambda value")

    poly_opt = GradientDescent(poly_model, alpha=0.1)
    for l in lambda_vec:
        poly_model.reg = l
        poly_opt.optimize_full_batch(utils.add_one(X1_poly_normalize), y_1)
        plot_fit(X_1, poly_model, plt, X1_poly_mu, X1_poly_sigma)

    plt.plot(X_1, y_1, 'rx')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title("Polynomial curve for different lambda values")
    plt.legend(lambda_vec)
    plt.savefig("img/ex6-fit.pdf")
    plt.show()

    # ## Polynomial logistic regression - regularized

    data = np.genfromtxt('data2.txt', delimiter=",", dtype="float")
    X = data[:, 0:2]
    y = data[:, 2]

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y[:])
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    plt.savefig("img/ex6-data2.pdf")
    plt.show()

    poly_X = map_feature(X[:, 0], X[:, 1])
    logReg = LogisticRegression(poly_X.shape[1])
    logReg_opt = GradientDescent(logReg, alpha=20)

    lambda_arr = [0, .01, .05, .1, .2, .5, 1]

    for i, l in enumerate(lambda_arr):
        logReg.reg = l
        fig, ax = plt.subplots()
        ax.set_title("Lambda: " + str(lambda_arr[i]))
        ax.set_xlabel('Microchip Test 1')
        ax.set_ylabel('Microchip Test 2')
        ax.scatter(X[:, 0], X[:, 1], c=y[:])
        logReg_opt.optimize_full_batch(poly_X, y)
        plot_decision_boundary(logReg.theta, ax)
        plt.savefig(f"img/DB-lambda_{l}.pdf")
        fig.show()

    # Compute accuracy on our training set
    logReg.reg = 0
    logReg_opt.optimize_full_batch(poly_X, y)
    p = logReg.predict(poly_X)
    perm = np.random.permutation(len(y))
    X_perm = poly_X[perm, :]
    y_perm = y[perm]
    y_pred_test = utils.cross_validation(X_perm, y_perm, 5, lambda model: GradientDescent(model, num_iters=1000, alpha=20), lambda: LogisticRegression(poly_X.shape[1]))

    print(f'Train Accuracy: {np.mean(p == y) * 100}%')

    print(f'Test Accuracy: {np.mean(y_pred_test == y_perm) * 100}%')

