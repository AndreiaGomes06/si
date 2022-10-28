# -*- coding: utf-8 -*-
import numpy as np

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    """
    Calculates the function sigmoid to X. It returns the probability of the values ​being equal to 1. Sigmoid algorithm: 1 / (1 + e**-X)
    Parameters
    ----------
    X: np.ndarray
        The input values
    """
    return 1 / (1 + np.exp(-x))