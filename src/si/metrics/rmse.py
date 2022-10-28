import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Returns the Root Mean Squared Error metric
    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    N = len(y_true)
    return np.sqrt(np.sum(np.subtract(y_true, y_pred)**2 / N))
    