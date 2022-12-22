import scipy.io
import utils
import numpy as np
from model import Perceptron
from optimize import GradientDescent
import matplotlib.pyplot as plt

input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10


# (note that we have mapped "0" to label 10)

def load_points(fn):
    mat = scipy.io.loadmat(fn)
    return mat["X"], mat["y"].reshape(-1) - 1


def load_weights(fn):
    mat = scipy.io.loadmat(fn)
    return utils.unroll_params([mat["Theta1"].T, mat["Theta2"].T])


def display_data(X, side=7):
    plt.figure()
    fig, axes = plt.subplots(side, side)

    i = 0
    for ax in axes.flatten():
        ax.imshow(np.reshape(X[i, :], [20, 20]).T)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        i += 1
    plt.savefig("data2.pdf")
    plt.show()

def check_gradient(X, y):
    classifier = Perceptron(X.shape[1] - 1, 10, 1)

    def num_cost(theta):
        classifier.theta = theta
        return classifier.cost(X, y)

    grad = classifier.grad(X, y)
    print(f'Gradient norm: {np.linalg.norm(grad)}')
    numgrad = utils.numgrad(num_cost, classifier.theta)
    norm = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-7). \n'
          f'\nRelative Difference: {norm}\n')

    classifier.reg = 1
    grad = classifier.grad(X, y)
    print(f'Gradient norm with regularization: {np.linalg.norm(grad)}')
    numgrad = utils.numgrad(num_cost, classifier.theta)
    norm = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-7). \n'
          f'\nRelative Difference: {norm}\n')


if __name__ == '__main__':
    X, y = load_points("data.mat")
    m, n = X.shape
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]
    display_data(X_perm)

    classifier = Perceptron(X.shape[1], num_labels, hidden_layer_size)
    theta = load_weights('weights.mat')
    Theta1 = np.reshape(theta[:classifier.theta1_size], classifier.theta1_shape)
    Theta1 = Theta1[:-1, :]
    display_data(Theta1.T, 5)

    X = utils.add_one(X)
    X_perm = utils.add_one(X_perm)
    y_perm = y[perm]
    max_index = m // 10 * 9
    X_train = X_perm[:max_index, :]
    X_test = X_perm[max_index:, :]
    y_train = y_perm[:max_index]
    y_test = y_perm[max_index:]
    temp = classifier.theta
    classifier.theta = theta
    J = classifier.cost(X, y)
    print(f'Cost at parameters (loaded from ex4weights): {J} '
          '\n(this value should be about 0.287629)\n')

    classifier.reg = 1
    J = classifier.cost(X, y)
    print(f'Cost at parameters (loaded from ex4weights): {J} '
          '\n(this value should be about 0.383770)\n')
    check_gradient(X, y)
    opt = GradientDescent(classifier, alpha=.3)
    print("Training classifier...")
    classifier.theta = temp
    opt.optimize_full_batch(X_train, y_train)
    print("Done.")
    Theta1 = np.reshape(classifier.theta[:classifier.theta1_size], classifier.theta1_shape)
    Theta1 = Theta1[:-1, :]
    display_data(Theta1.T, 5)

    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)

    print("Train accuracy: %d%%" % (100 * np.sum(y_train == pred_train) / len(y_train)))
    print("Test accuracy: %d%%" % (100 * np.sum(y_test == pred_test) / len(y_test)))
