from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import joblib

class LogisticClassifier:
    def __init__(self, cs=10, cv=5, random_state=42,max_iter=1e4):
        self.classifier = LogisticRegressionCV(Cs=cs, cv=cv, random_state=random_state,max_iter=max_iter)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.classifier.predict(X)
        
    def score(self, X, y):
        X = self.scaler.transform(X)
        return self.classifier.score(X, y)

    def save(self, filename):
        joblib.dump((self.classifier, self.scaler), filename)

    @classmethod
    def load(cls, filename):
        classifier, scaler = joblib.load(filename)
        instance = cls()
        instance.classifier = classifier
        instance.scaler = scaler
        return instance
    