
from utils import build_dict, transform, cross_validation, normalize_features
from optimize import GradientDescent
from model import LinearRegression
import numpy as np

def test():
    mapping = build_dict(["first", "second", "first", "third", "first", "second"])
    print(transform(mapping, ["first", "second", "first", "fourth", "fifth"]))

def load_data(path):
    data = []
    with open(path) as f:
        for line in f.read().splitlines():
            data.append(line.split(","))
    data = transform_features(data)
    return np.concatenate([np.ones([data.shape[0], 1]), data[:, :-1]], axis=1), data[:, -1]

def transform_features(X):
    X_a = np.array(X)
    mapping1 = build_dict(X_a[:, 0])
    mapping2 = build_dict(X_a[:, 1])
    X_trans = transform(mapping1, X_a[:, 0])[:, 1:]
    # X_trans = np.concatenate([X_trans, transform(mapping2, X_a[:, 1])[:, 1:]], axis=1)
    for i in range(2, len(X[0])):
        X_trans = np.concatenate([X_trans, np.expand_dims(np.array(X_a[:, i]).astype(float), axis=1)], axis=1)
    X_trans[:, :-1], mu, sigma = normalize_features(X_trans[:, :-1])
    return X_trans


def eval(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mre = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    print("Absolute error: %f" % mae)
    print("Relative error: %f%%" % (mre * 100))
    return mae, mre

if __name__ == '__main__':
    test()
    X, y = load_data("data_machines.txt")
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]
    y_perm = y[perm]
    y_pred_test = cross_validation(X_perm, y_perm, 10, lambda model: GradientDescent(model, num_iters=1000, alpha=0.1), lambda: LinearRegression(X.shape[1]))
    print("Test eval:")
    eval(y_perm, y_pred_test)
    linReg = LinearRegression(X.shape[1])
    opt = GradientDescent(linReg, num_iters=1000, alpha=0.1)
    opt.optimize_full_batch(X_perm, y_perm)
    y_pred_train = linReg.predict(X_perm)
    print("Train eval:")
    eval(y_perm, y_pred_train)
