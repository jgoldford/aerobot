from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import joblib
import torch
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
from tqdm import tqdm
from typing import Tuple, NoReturn, List, Dict
from sklearn.linear_model import LogisticRegression
import pandas as pd
import inspect


# Use a GPU if one is available. 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Nonlinear(torch.nn.Module):
    '''Two-layer neural network for classification.'''
    def __init__(self, input_dim:int=None, 
        weight_decay:float=0.01, 
        hidden_dim:int=512, 
        lr:float=0.00001, 
        n_epochs:int=1000, 
        batch_size:int=16, 
        alpha:int=10, 
        early_stopping:bool=True, 
        n_classes:int=3):
        '''Initialize a Nonlinear classifier.

        :param input_dim: The dimensionality of input vectors. This is the number of nodes in the first linear layer.
        :param weight_decay: The L2 regularization penalty to be passed into the Adam optimizer.
        :param hidden_dim: The number of nodes in the second linear layer. 
        :param lr: The learning rate.
        :param n_epochs: The maximum number of epochs to train the classifier. 
        :param batch_size: The size of the batches for model training. 
        :param alpha: The early stopping threshold. Early stopping is triggered when the generalization loss exceeds this threshold,
            and early_stopping=True.
        :param early_stopping: Whether or not to use early stopping during training. 
        :param n_classes: The number of classes. This is the output dimension of the second linear layer. 
        '''        
        torch.manual_seed(42) # Seed the RNG for reproducibility.
        super().__init__()

        self.val_accs, self.train_accs, self.train_losses, self.val_losses = [], [], [], []
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        self.classes_ = None # Will be populated later, for consistency with LogisticRegression model.
        self.n_classes = n_classes
        self.encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='error', sparse_output=False)
        self.lr = lr
        self.alpha = alpha
        self.early_stopping = early_stopping
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.n_classes)).to(device)

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)


    def score(self, X:np.ndarray, y:np.ndarray):
        pass

    def _get_batches(self, X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Create batches of size batch_size from training data and labels.'''
        # Don't bother with balanced batches. Doesn't help much with accuracy anyway. 
        n_batches = len(X) // self.batch_size + 1
        X_batches = np.array_split(X, n_batches, axis=0)
        y_batches = np.array_split(y, n_batches, axis=0)
        return X_batches, y_batches

    @staticmethod
    def shuffle(X:np.ndarray, y:np.ndarray):
        shuffle_idxs = np.arange(len(X)) # Get indices to shuffle the inputs and labels. 
        np.random.shuffle(shuffle_idxs)
        X = X[shuffle_idxs, :]
        y = y[shuffle_idxs, :]
        return X, y

    def forward(self, X:np.ndarray) -> torch.FloatTensor:
        '''A forward pass of the model.'''
        X = torch.FloatTensor(X).to(device) # Convert numpy array to Tensor. Make sure it is on the GPU, if available.
        return self.classifier(X)

    def _early_stop(self, t:int):
        '''The GL early-stopping criterion from https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf.'''

        if (not self.early_stopping) or (len(self.val_losses) == 0):
            return False

        def generalization_error():
            '''Compute the generalization error using the best validation loss obtained so far and the current
            validation loss.'''
            min_val_loss = min(self.val_losses) # Get the smallest validation loss from the stored history.
            curr_val_loss = self.val_losses[-1] # Get the most recent validation loss from the stored history. 
            return 100 * ((curr_val_loss / min_val_loss) - 1)

        return generalization_error() > self.alpha
        
    def balanced_accuracy(self, X:np.ndarray, y:np.ndarray):
        y_pred = self.predict(X)
        return balanced_accuracy_score(y.ravel(), y_pred.ravel())
    
    def loss_func(self, y_pred:torch.FloatTensor, y:torch.FloatTensor, weight:torch.FloatTensor=None):
        '''Implement the loss function specified on initialization. The addition of this function wrapper allows weights
        to be easily discarded if mean-squared error is used.
        
        :param y_pred: A torch FloatTensor of shape (n, self.n_classes), where n is the number of elements in the batch, dataset, etc.
        :param y: A torch FloatTensor of shape (n, self.n_classes), where n is the number of elements in the batch, dataset, etc.
        '''
        weight = torch.FloatTensor([1, 1, 1]) if weight is None else weight
        # Make sure to apply the softmax, which is done automatically when using cross-entropy loss. 
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        # y_pred = (y * y_pred).sum(axis=1, keepdims=True) # Only keep prediction elements which are nonzero by treating target array like a mask. 
        # Get a 16-dimensional column vector of weights for each row. 
        weight = torch.matmul(y, weight.reshape(self.n_classes, 1))
        return torch.mean((y - y_pred)**2 * weight) 

    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray=None, y_val:np.ndarray=None, verbose:bool=True):
        '''Train the nonlinear classifier on input data.'''

        y_enc = self.encoder.fit_transform(y.reshape(-1, 1)) # One-hot encode the training targets. 
        y_val_enc = y_val if (y_val is None) else self.encoder.transform(y_val.reshape(-1, 1)) 
        
        # self.classes_ = self.encoder.categories_[0] # Extract categories from the one-hot encoder. 
        self.classes_ = self.encoder.categories_[0] # Extract categories from the one-hot encoder. 
        self.weight = torch.FloatTensor([1 / (np.sum(y == c) / len(y)) for c in self.classes_]) # Compute loss weights as the inverse frequency.

        self.train() # Model in train mode.  
        for epoch in tqdm(range(self.n_epochs), desc='Training NonlinearClassifier...', disable=not verbose):
            X_trans, y_trans = Nonlinear.shuffle(X, y_enc) # Shuffle the transformed data. 
            X_batches, y_batches = self._get_batches(X, y_enc) 
            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred = self(X_batch)
                train_loss = self.loss_func(y_pred, torch.FloatTensor(y_batch).to(device), weight=self.weight)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
              
            self.train_losses.append(self.loss_func(self(X), torch.FloatTensor(y_enc), weight=self.weight).item()) # Store the average weighted train losses over the epoch. 
            self.train_accs.append(self.balanced_accuracy(X, y)) # Store model accuracy on the training dataset. 
            if (X_val is not None) and (y_val is not None):
                self.val_losses.append(self.loss_func(self(X_val), torch.FloatTensor(y_val_enc)).item()) # Store the unweighted loss on the validation data.
                self.val_accs.append(self.balanced_accuracy(X_val, y_val)) # Store model accuracy on the validation dataset. 

            if self._early_stop(epoch):
                # self.classifier.load_state_dict(previous_state_dict) # Load the previous state dict.
                self.n_epochs = epoch # Overwrite the previously-set number of epochs. 
                if verbose: print(f'NonlinearClassifier.fit: Terminated training at epoch {epoch}.')
                break

            self.train() # Make sure to put the model back into training mode, as the predict function switches to evaluation mode. 

    def predict(self, X:np.ndarray, label_output:bool=True) -> np.ndarray:
        self.eval() # Model in evaluation mode.
        output = self(X)

        def parse_output(output:np.ndarray) -> np.ndarray:
            # Softmax is included in loss function, so needs to be applied here if output is not being fed into self.loss_func.
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.detach().numpy() # Convert output tensor to numpy array. 
            y_pred = np.zeros(output.shape)
            # Find the maximum output in the three-value output, and set it to 1. Basically converting to a one-hot encoding.
            for i in range(len(output)):
                j = np.argmax(output[i])
                y_pred[i, j] = 1
            return self.encoder.inverse_transform(y_pred)
        
        return parse_output(output) if label_output else output


class RandomRelative():
    def __init__(self, tax_data:pd.DataFrame=None, level:str='Phylum'):
        '''Initialize a RandomRelative classifier.
        
        :param tax_data: A DataFrame containing taxonomy information for each genome and their physiologies.
        :param level: The taxonomic rank to use for mean-relative classification. 
        '''
        self.levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.
        assert level in self.levels, f'RandomRelative.__init__: Specified level {level} is invalid.' 
        assert len(set(self.levels).intersection(set(tax_data.columns))) == len(self.levels), 'RandomRelative.__init__: Taxonomy columns are missing from tax_data.'
        assert 'physiology' in tax_data.columns, 'RandomRelative.__init__: Physiology label is missing from tax_data.'
        self.tax_data = tax_data[self.levels + ['physiology']] # Make sure only necessary columns are included.
        self.level = level
        self.classes_ = np.unique(tax_data.physiology.values)

    def fit(self, X:np.ndarray, y:np.ndarray):
        '''For compatibility, does not actually do anything.'''
        pass

    def predict(self, X:np.ndarray):
        # Get the taxonomy label at self.level for each genome ID in X.
        tax = self.tax_data.loc[X.ravel(), self.level].values
        y_pred = []
        for t in tax:
            relatives = self.tax_data[self.tax_data[self.level] == t] # Get all relatives at the specified level.
            y_pred.append(np.random.choice(relatives.physiology.values)) # Choose a random physiology label from among the relatives. 
        y_pred = np.array(y_pred).ravel()
        return y_pred


class MeanRelative():
    '''Classifier which makes a metabolic prediction based on the average of the metabolisms of the closest relatives.'''
    def __init__(self, tax_data:pd.DataFrame=None, level:str='Phylum'):
        '''Initialize a MeanRelative classifier.
        
        :param tax_data: A DataFrame containing taxonomy information for each genome and their physiologies.
        :param level: The taxonomic rank to use for mean-relative classification. 
        '''
        self.levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.
        assert level in self.levels, f'RandomRelative.__init__: Specified level {level} is invalid.' 
        assert len(set(self.levels).intersection(set(tax_data.columns))) == len(self.levels), 'RandomRelative.__init__: Taxonomy columns are missing from tax_data.'
        assert 'physiology' in tax_data.columns, 'RandomRelative.__init__: Physiology label is missing from tax_data.'
        self.tax_data = tax_data[self.levels + ['physiology']] # Make sure only necessary columns are included.

        self.encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='error', sparse_output=False)
        self.encoder.fit(tax_data.physiology.values.reshape(-1, 1)) # Fit the encoder. 
        self.classes_ = self.encoder.categories_[0] # Srore the number of categories. 
        self.n_classes = len(self.classes_) # Store the number of classes.
        self.level = level

        self.means = self._get_means()

    def _get_means(self) -> Dict[str, str]:
        '''Create a dictionary matching each taxonomical category to the "mean" metabolic prediction. Mean predictions are 
        generated by one-hot encoding each physiology label, and averaging across each member of a taxonomic group. Then, 
        the maximum value in the resulting array is taken to be the "mean" category, and is converted back to a physiology
        label using the fitted one-hot encoder.'''
        means = dict()
        for tax, relatives in self.tax_data.groupby(self.level):
            physiologies = relatives.physiology.values.reshape(-1, 1) # Reshape for the encoder.
            physiologies = self.encoder.transform(physiologies) # Should be an n-by-3 array (or 2 for binary classification).
            mean = np.mean(physiologies, axis=0).ravel() # Want to take the average of the rows. 
            assert len(mean) == self.n_classes, 'MeanRelative._get_means: Something went wrong with calculating mean physiology.'
            mean = np.array([[1 if i == np.argmax(mean) else 0 for i in range(self.n_classes)]]) # Convert mean to an array of 1's and 0's for inverse one-hot encoding.
            means[tax] = self.encoder.inverse_transform(mean)[0][0] # Store the physiology prediction in the dictionary
        return means

    def fit(self, X:np.ndarray, y:np.ndarray):
        '''For compatibility, does not actually do anything.'''
        pass

    def predict(self, X:np.ndarray):
        '''
        :param X: An array of strings, all of which are genome IDs.
        :paray y: An array of strings indicating the correct physiology labels.
        '''
        # Get the taxonomy label at self.level for each genome ID in X.
        tax = self.tax_data.loc[X.ravel(), self.level].values
        y_pred = np.array([self.means[t] for t in tax]) # Get the pre-computed mean predictions.
        return y_pred


class GeneralClassifier():
    def __init__(self, model_class, params:Dict[str, object]=None, normalize:bool=True):
        '''Initialization function for a general classifier.

        :param model_class: Model class to fit. The class must implement the fit, predict, and score methods. 
        :param params: A dictionary of parameters to pass into the model_class instantiator.
        :param normalize: Whether or not to standardize the input data.
        '''
        self.classifier = model_class(**params) if params else model_class()
        self.scaler = StandardScaler() if normalize else None
        self.n_classes = None # To be populated after fitting the model, indicated binary or ternary classification. 

    def fit(self, X:np.ndarray, y:np.ndarray, X_val:np.ndarray=None, y_val:np.ndarray=None):
        '''Fit the underlying model to training data.'''
        X = X if (not self.scaler) else self.scaler.fit_transform(X) # Standardize the input, if specified.

        self.n_classes = len(np.unique(y)) # Get the number of classes. 

        if (X_val is not None) and (y_val is not None):
            X_val = X_val if (not self.scaler) else self.scaler.fit_transform(X_val)
            self.classifier.fit(X, y, X_val=X_val, y_val=y_val)
        else:
            self.classifier.fit(X, y)

    def predict(self, X:np.ndarray) -> np.ndarray:
        X = X if (not self.scaler) else self.scaler.transform(X) # Standardize the input, if specified.
        return self.classifier.predict(X)
        
    def score(self, X:np.ndarray, y:np.ndarray):
        X = X if (not self.scaler) else self.scaler.fit_transform(X) # Standardize the input, if specified.
        return self.classifier.score(X, y)
    
    def balanced_accuracy(self, X:np.ndarray, y:np.ndarray) -> float:
        X = X if (not self.scaler) else self.scaler.transform(X) # Standardize the input, if specified.
        y_pred = self.classifier.predict(X)
        return balanced_accuracy_score(y, y_pred)

    def confusion_matrix(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        X = X if (not self.scaler) else self.scaler.transform(X) # Standardize the input, if specified.
        y_pred = self.classifier.predict(X)
        return confusion_matrix(y, y_pred)

    def f1_score(self, X:np.ndarray, y:np.ndarray) -> float:
        '''Compute the F1 score, which is defined as tp / (tp + 0.5 * (fp + fn)), or 2 / (1/precition + 1/recall) for a binary classification problem.
        For a ternary classification problem, use a one-versus-all approach to compute the individual scores for each class, and
        then taking the average (which can be weighted).'''
        X = X if (not self.scaler) else self.scaler.transform(X) # Standardize the input, if specified.
        y_pred = self.classifier.predict(X)
        return f1_score(y, y_pred, average='weighted' if self.n_classes > 2 else 'binary')

    def save(self, path:str) -> NoReturn:
        '''Save the GeneralClassifier instance to a file.

        :param path: The location where the object will be stored.
        '''
        joblib.dump((self.classifier, self.scaler), path)

    @classmethod
    def load(cls, path:str):
        '''Load a saved GeneralClassifier object from a file.

        :param path: The path to the file where the object is stored.
        :return: A GeneralClassifier instance.
        '''
        classifier, scaler = joblib.load(path)
        # Need to pass keyword arguments for the Nonlinear classifier.
        if isinstance(classifier, Nonlinear):
            kwargs = ['input_dim', 'hidden_dim'] # Arguments required for initializer to not throw an error. 
            params = {k:getattr(classifier, k) for k in kwargs}
            instance = cls(model_class=type(classifier), params=params)
        elif isinstance(classifier, LogisticRegression):
            instance = cls(model_class=type(classifier))
        instance.classifier = classifier
        instance.scaler = scaler
        return instance


def evaluate(model:GeneralClassifier, X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray) -> Dict:
    '''Evaluate a trained GeneralClassifier using the training and test data.

    :param model: The trained GeneralClassifier to evaluate.
    :param X: A numpy array containing the training features.
    :param y: A numpy array containing the training labels.
    :param X_val: A numpy array containing the validation features.
    :param y_val: A numpy array containing the validation labels. 
    :return: A dictionary containing various evaluation metrics for the trained classifier on the training
        and validation data.
    '''
    results = dict()

    results['training_acc'] = model.balanced_accuracy(X, y)
    results['validation_acc'] = model.balanced_accuracy(X_val, y_val)
    results['classes'] = model.classifier.classes_
    results['confusion_matrix'] = model.confusion_matrix(X_val, y_val).ravel()
    results['f1_score'] = model.f1_score(X_val, y_val)

    if isinstance(model.classifier, LogisticRegression): # Only applies if the model used LogisticRegression.
        n_iter = model.classifier.n_iter_[0]
        results['n_iter'] = n_iter
        results['converged'] = n_iter < model.classifier.max_iter

    if isinstance(model.classifier, Nonlinear): # If the underlying model is Nonlinear...
        # Save some information for plotting training curves.
        results['training_losses'] = model.classifier.train_losses
        results['validation_losses'] = model.classifier.val_losses
        results['training_accs'] = model.classifier.train_accs
        results['validation_accs'] = model.classifier.val_accs
    
    return results


 

