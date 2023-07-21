from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit


#from sklearn import metrics
import joblib


class LogisticClassifier:
    def __init__(self, cs=10, cv=5, random_state=42, max_iter=10000, normalize=True):
        self.classifier = LogisticRegressionCV(Cs=cs, cv=cv, random_state=random_state, max_iter=max_iter)
        self.scaler = StandardScaler() if normalize else None

    def fit(self, X, y):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.classifier.predict(X)
        
    def score(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.classifier.score(X, y)
    
    def balanced_accuracy(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return balanced_accuracy_score(y, y_pred)

    def confusion_matrix(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return confusion_matrix(y, y_pred)

    def save(self, filename):
        joblib.dump((self.classifier, self.scaler), filename)

    @classmethod
    def load(cls, filename):
        classifier, scaler = joblib.load(filename)
        instance = cls()
        instance.classifier = classifier
        instance.scaler = scaler
        return instance

class GeneralClassifier:
    def __init__(self, model_class, params=None, normalize=True):
        self.classifier = model_class(**params) if params else model_class()
        self.scaler = StandardScaler() if normalize else None

    def fit(self, X, y):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.classifier.predict(X)
        
    def score(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.classifier.score(X, y)
    
    def balanced_accuracy(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return balanced_accuracy_score(y, y_pred)

    def confusion_matrix(self, X, y):
        if self.scaler:
            X = self.scaler.transform(X)
        y_pred = self.classifier.predict(X)
        return confusion_matrix(y, y_pred)

    def hyperparameter_optimization(self, X, y, param_grid, cv_strategy=None):
        if self.scaler:
            X = self.scaler.fit_transform(X)

        # Use StratifiedShuffleSplit as the default CV strategy if not provided
        if not cv_strategy:
            cv_strategy = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

        grid_search = GridSearchCV(self.classifier, param_grid, cv=cv_strategy)
        grid_search.fit(X, y)
        self.classifier = grid_search.best_estimator_
        return grid_search.best_params_

    def perform_cross_validation(self, X, y, cv=5):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        scores = cross_val_score(self.classifier, X, y, cv=cv)
        return scores.mean(), scores.std()

    def save(self, filename):
        joblib.dump((self.classifier, self.scaler), filename)

    @classmethod
    def load(cls, filename):
        classifier, scaler = joblib.load(filename)
        instance = cls(model_class=type(classifier))
        instance.classifier = classifier
        instance.scaler = scaler
        return instance
