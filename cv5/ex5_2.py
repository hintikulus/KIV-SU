import scipy.io
import utils
import numpy as np
from model import LogisticRegression, OneVsAll
from optimize import GradientDescent
import matplotlib.pyplot as plt
import visualize as vis

def load_points(fn):
    mat = scipy.io.loadmat(fn)
    return mat["X"], mat["y"].reshape(-1) - 1

def display_data(X):
    plt.figure()
    fig, axes = plt.subplots(7, 7)

    i = 0
    for ax in axes.flatten():
        ax.imshow(np.reshape(X[i, :], [20, 20]).T)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        i += 1
    plt.show()


if __name__ == '__main__':
    X, y = load_points("data2.mat")
    m, n = X.shape
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]
    display_data(X_perm)
    X = utils.add_one(X)
    X_perm = utils.add_one(X_perm)
    y_perm = y[perm]
    max_index = m // 10 * 9
    X_train = X_perm[:max_index, :]
    X_test = X_perm[max_index:, :]
    y_train = y_perm[:max_index]
    y_test = y_perm[max_index:]
    classifier = OneVsAll(lambda num_features: LogisticRegression(num_features), lambda model: GradientDescent(model, alpha=.1, num_iters=400))
    print("Training One-vs-all classifier...")
    classifier.train(X_train, y_train)
    print("Done.")

    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)

    print("Train accuracy: %d%%" % (100 * np.sum(y_train == pred_train) / len(y_train)))
    print("Test accuracy: %d%%" % (100 * np.sum(y_test == pred_test) / len(y_test)))

