import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class CAV:
    def __init__(self, concept_name, model_name, vector):
        self.concept_name = concept_name
        self.model_name = model_name
        self.vector = vector

def train_cav(concept_activations, random_activations):
    """
    Trains a linear classifier to separate concept activations from random examples.
    Returns the normal vector of the decision boundary (CAV).
    """
    X = np.concatenate([concept_activations, random_activations])
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)
    
    return clf.coef_.flatten()  # The CAV
