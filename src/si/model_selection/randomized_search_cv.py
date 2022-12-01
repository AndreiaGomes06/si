# -*- coding: utf-8 -*-
import itertools 
from typing import Callable, Dict, List, Tuple, Union

import random
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from model_selection.cross_validate import cross_validate


def randomized_search_cv(model, dataset: Dataset, parameter_distribution: Union[Dict, List[Dict]], scoring: Callable = None, cv: int = 3, n_iter: int = 2, test_size: float = 0.2):
    """
    Performs a randomized search cross validation on a model.
    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    parameter_distribution: Union[Dict, List[Dict]] 
        The parameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter: int
        Number of iterations.
    test_size: float
        The test size.
    """
    # validate the parameter grid
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}")
        
    scores = []
    parameterslist =[]

    for n in range(n_iter):
    # for each combination
        dic = {}
        for parameter in parameter_distribution:
            dic[parameter] = np.random.choice(parameter_distribution[parameter])
 
        parameterslist.append(dic)

    for combination in parameterslist:
        parameters = {} 
        # set the parameters
        for parameter, value in zip(combination.keys(), combination.values()):
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
    from sklearn.preprocessing import StandardScaler

    # load and split the dataset
    sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\io')
    from csv1 import read_csv
    path = r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)
    breast.X = StandardScaler().fit_transform(breast.X) 

    # initialize the Logistic Regression model
    knn = LogisticRegression(use_adaptive_alpha = False)

    # parameter grid
    parameter_distribution = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200, dtype = int)
    }

    # cross validate the model
    scores_ = randomized_search_cv(knn,
                             breast,
                             parameter_distribution=parameter_distribution,
                             cv = 3,
                             n_iter = 10)

    # print the scores
    print(f"Scores: {scores_}\n") 