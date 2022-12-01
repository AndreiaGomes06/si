# -*- coding: utf-8 -*-
import itertools 
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from model_selection.cross_validate import cross_validate

def grid_search_cv(model, dataset: Dataset, parameter_grid: Dict[str, Tuple], scoring: Callable = None, cv: int = 3, test_size: float = 0.2,) -> Dict[str, List[float]]:
    """
    Performs a grid search cross validation on a model.
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    parameter_grid: Dict[str, Tuple]  
        The parameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    """
    # validate the parameter grid
    for parameter in parameter_grid: 
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    scores = []

    # for each combination
    for combination in itertools.product(*parameter_grid.values()): 
        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination): 
            setattr(model, parameter, value) 
            parameters[parameter] = value

        # cross validate the model 
        score = cross_validate(model = model, dataset = dataset, scoring = scoring, cv = cv, test_size = test_size)

        # add the parameter configuration
        score['parameters'] = parameters

        # add the score
        scores.append(score)

    return scores


if __name__ == '__main__':
    print("Example 1")
    # import dataset
    from data.dataset import Dataset
    from linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression(use_adaptive_alpha = False)

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    scores_ = grid_search_cv(knn,
                             dataset_,
                             parameter_grid=parameter_grid_,
                             cv = 3)

    # print the scores
    print(f"Scores: {scores_}\n") 

    print("Example 2")
    sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\io')
    from csv1 import read_csv
    path = r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)
    
    from sklearn.preprocessing import StandardScaler
    breast.X = StandardScaler().fit_transform(breast.X) 
    
    reg = LogisticRegression(use_adaptive_alpha=False)
    parameter_grid = {'l2_penalty': (1, 10), 'alpha': (0.001, 0.0001), 'max_iter': (1000, 2000)}
    scores = grid_search_cv(reg, breast, parameter_grid, cv=3)
    print(f"Scores: {scores}") 