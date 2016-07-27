#!/usr/bin/env python
# coding=utf-8
"""
2016  6,18 , huangpingping
Brief: logistic regression for minist data  set
"""

import os
import sys
import numpy as np

def load_dataset(fp):
    """ load mnist data forom file path fp
    except:
        IOException
    """
    fin = open(fp)
    array =  np.loadtxt(fin, delimiter=",", dtype=np.float, unpack=False)
    fin.close()
    labs = array[:, 0]
    feas = array[:, 1:]
    return labs, feas, feas.shape[0]


def hypothesis(X, W):
    """ hypothesis function using one-vs-all strategy for lr
    args:
        X: data set
        W: hypothesis weights
    returns:
        column-wise matrix of probabilities
    """
    m, n = X.shape
    W = W.reshape((n, n_class))
    sigmoid = 1 / (1 + np.exp(-np.dot(X, W)))
    return sigmoid


def cost(X, Y, W):
    """ cost function for lr
    args:
        X: data set
        Y: target set
        W: hypothesis weights
    returns:
        a scalar value of cost
    """
    m, n = X.shape
    h = hypothesis(X, W)
    residuals = Y * np.log(h) + (1 - Y) * np.log(1 - h)
    loss = residuals.sum() / -m
    return loss


def gradient(X, Y, W):
    """ gradient for lr losss without regulariztion terms
    args:
        X: data feature matrix 
        Y: target data matrix
        W: hypothesis weights
    return:
        gradient matrix for $(W)
    """
    m, n = X.shape
    h = hypothesis(X, W)
    grad = np.dot(X.T, h - Y) / m
    return grad.reshape(grad.size)


def evaluate(info, X, Y, L, W):
    """ evaluate performance
    args:
        info: text message
        X: data feature matrix 
        Y: target data matrix
        L: gold standard labels matrix
        W: hypothesis weights
    """
    n = len(X)
    j = cost(X, Y, W)  

    h = np.argmax(hypothesis(X, W), axis=1)  # get predict values
    incorrect = (h != L).sum()
    success = (n - incorrect + 0.0) / n

    print(info, "incorrect=%i, errorRate:%.1f%%, accuracy=%.1f%% (%d /%d), cost=%.8f"\
          % (incorrect, incorrect * 100.0 / n, success * 100, (n - incorrect), n, j))


def gd_optimize(X, Y, W, N, alpha):
    """ gradient decent optimization
    """
    for i in range(N):
        cost_i = cost(X, Y, W)
        gra = gradient(X, Y, W)
        W -= gra * alpha
        sys.stderr.write("Ite\t%d\tcost\t%.2f\n" % (i, cost_i))


def train(train_f):
    """ build the linear model and optimize weights with gradient decenting
    args:
        train_f: the input train f of mnist data
    returns:
        weights
    """
    N = 100  #number of iteration
    alpha = 1  #learning rate
    
    t_labs, t_feas, n_cnt = load_dataset(train_f)

    # add bias vector 
    X = np.hstack((np.ones((n_cnt, 1), dtype = np.float64), t_feas / 255.0))
    Y = np.zeros((n_cnt, n_class), dtype = np.float64)
    Y[range(n_cnt), t_labs.astype(int)] = 1

    # initial weights
    weights = np.zeros((image_m * image_n + 1) * n_class)

    gd_optimize(X, Y, weights, N, alpha)

    evaluate(">> Performance on training set", X, Y, t_labs, weights)

    return weights


def test(test_f, weights):
    """ test process
    """
    test_labs, test_feas, test_cnt= load_dataset(test_f)

    X = np.hstack((np.ones((test_cnt, 1), dtype=np.float64), test_feas / 255.0))
    Y = np.zeros((test_cnt, n_class), dtype=np.float64)
    Y[range(test_cnt), test_labs.astype(int)] = 1

    evaluate(">> Performace on test set", X, Y, test_labs, weights)


n_class = 10 # number of classes
image_n, image_m = 28, 28 # size of image 

if __name__ == "__main__":
    data_dir = "../data"
    valid_f = os.path.join(data_dir, "validation") 
    train_f = os.path.join(data_dir, "training")
    test_f = os.path.join(data_dir, "testing")
    sys.stderr.write(">> Begin training\n")

    weights = train(train_f)
    
    test(test_f, weights)
