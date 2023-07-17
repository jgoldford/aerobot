from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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