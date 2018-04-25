import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    # YOUR CODE HERE:
    X_out = np.ndarray(shape=np.shape(X))

    mean = np.mean(X, axis=0)
    deviation = np.std(X, axis=0)

    X_out = (X - mean)/deviation
    # END YOUR CODE

    return X_out


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """

    # YOUR CODE HERE:
    error = np.dot(X, w) - y
    error_trans = np.transpose(error)
    J = (np.dot(error_trans, error))/np.shape(y)[0]
    J = J[0][0]

    # END YOUR CODE

    return J


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    # YOUR CODE HERE:
    errors = np.dot(X, w) - y

    X_trans = np.transpose(X)
    grad = np.dot(X_trans, errors)
    grad *= (2/np.shape(X)[0])

    # END YOUR CODE

    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]

    # YOUR CODE HERE:
    for i in range (0, num_iters):
        w = w - (learning_rate * compute_wgrad(X, y, w))
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))
    # END YOUR CODE

    return w, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    # YOUR CODE HERE:
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    dataset = np.concatenate((X, y), axis=1)

    for i in range (0, num_iters):
        np.random.shuffle(dataset)
        sample = dataset[:batch_size, :2]
        results = dataset[:batch_size, 2:]
        w = w - (learning_rate * compute_wgrad(sample, results, w))
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(sample, results, w))
    # END YOUR CODE

    return w, weights_history, cost_history
