import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the cross-entropy loss function.
    It's calculated by the difference of two probabilities, the true values and the predicted ones.
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset.
    y_pred: np.ndarray
        The predicted labels of the dataset.
    """
    return - np.sum(y_true * np.log(y_pred)) / len(y_true)


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the derivative of the cross-entropy loss function.
    It's calculated by the difference of two probabilities, the true values and the predicted ones.
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset.
    y_pred: np.ndarray
        The predicted labels of the dataset.
    """
    return y_pred - y_true   #derivada da função anterior-verificar