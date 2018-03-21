#9793735

import numpy as np

def normal_equation_prediction(x_dataset, y_dataset):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param x_dataset: design matrix
    :type x_dataset: np.array
    :param y_dataset: regression targets
    :type y_dataset: np.array
    :return: prediction
    :rtype: np.array

    """
    ones_line = np.ones((x_dataset.shape[0], 1))
    x_line = np.append(ones_line, x_dataset, axis=1)
    x_transpose = np.transpose(x_line)
    w_estimative = np.linalg.inv(np.dot(x_transpose, x_line))
    w_estimative = np.dot(w_estimative, x_transpose)
    w_estimative = np.dot(w_estimative, y_dataset)
    prediction = np.dot(x_line, w_estimative)

    return prediction
