from sklearn.base import BaseEstimator, TransformerMixin

class Selector(BaseEstimator, TransformerMixin):
    """
        Transformer to select a single column from the data frame to perform additional transformations on
        Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

        
    def fit(self, X, y=None):
        return self

    
    def transform(self, X):
        return X[self.key]
