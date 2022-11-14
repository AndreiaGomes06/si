# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
from data.dataset import Dataset
from metrics.accuracy import accuracy

class StackingClassifier:
    """
    Ensemble classifier that uses the results of a set of models as features of a final model.
    """
    def __init__(self, models, final_model): #[knn, lg], é passada um lista com os modelos de predição
        """
        Initialize the ensemble classifier.
        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        final_model:
            Final model classifier
        """
        # parameters
        self.models = models 
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.
        It trains the inicial models and uses their prdiction as features to train the final model
        Parameters
        ----------
        dataset : Dataset
            The training data.
        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset) #treinas os modelos

        for model in self.models:
            features = np.array([model.predict(dataset)]).transpose() 
        self.final_model.fit(Dataset((features), dataset.y))
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.
        The final model uses the other models predictions as features to get to the final prediction
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """
        for model in self.models:
            features = np.array([model.predict(dataset)]).transpose() 
            return self.final_model.predict(Dataset((features), dataset.y))
    
    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels for the final model.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))



if __name__ == '__main__':
    # import dataset
    import sys
    sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si')
    sys.path.insert(0, r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\src\si\io')

    from sklearn.preprocessing import StandardScaler
    from neighbors.knn_classifier import KNNClassifier
    from linear_model.logistic_regression import LogisticRegression
    from model_selection.split import train_test_split
    from csv1 import read_csv
    import numpy as np
    path = r'C:\Users\Asus\Desktop\Bioinformática\2º Ano\1º Semestre\Sistemas inteligentes\si\datasets\breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)

    breast.X = StandardScaler().fit_transform(breast.X) # para normalizar as features
    train, test = train_test_split(breast, test_size = 0.2, random_state = 8)
    
    knn = KNNClassifier(k = 3)
    lg = LogisticRegression(use_adaptive_alpha = False)
    final_model = KNNClassifier(k = 2)

    stacking = StackingClassifier([knn, lg], final_model)
    stacking.fit(train)
    print(f"Predictions: {stacking.predict(test)}") 
    print(f"Score: {stacking.score(test)}") 


