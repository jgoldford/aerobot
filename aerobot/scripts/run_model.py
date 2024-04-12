'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import numpy as np
import tqdm
from aerobot.io import ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES, NumpyEncoder
from aerobot.datasets import load_datasets
from sklearn.linear_model import LogisticRegression
from aerobot.models import GeneralClassifier, Nonlinear
# from joblib import Parallel, delayed
from os import path
import argparse
from typing import Dict, NoReturn
import pickle
import json


def train_logistic(X:np.ndarray, y:np.ndarray, params:Dict={'penalty':'l2', 'C':100, 'max_iter':10000}) -> GeneralClassifier:
    '''Train a LogisticRegression-based classifier.

    :param X: A numpy array containing the training features.
    :param y: A numpy array containing the training labels.
    :param params: A dictionary of keyword arguments to pass into the classifier initialization function.
    :return: A trained instance of a GeneralClassifier based on logistic regression.
    '''
    model = GeneralClassifier(model_class=LogisticRegression, params=params) # Instantiate a classifier.
    model.fit(X, y)
    return model


def train_nonlinear(X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray, params:Dict=None) -> GeneralClassifier:
    '''Train a Nonlinear-based classifier.

    :param X: A numpy array containing the training features.
    :param y: A numpy array containing the training labels.
    :param X_val: A numpy array containing the validation features.
    :param y_val: A numpy array containing the validation labels. 
    :param params: A dictionary of keyword arguments to pass into the classifier initialization function.
    :return: A trained instance of a GeneralClassifier based on a nonlinear neural network.
    '''
    model = GeneralClassifier(model_class=Nonlinear, params=params) # Instantiate a classifier.
    model.fit(X, y, X_val=X_val, y_val=y_val)
    return model


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


def print_summary(results:Dict) -> NoReturn:
    '''Print a summary of the evaluation results to the terminal.'''
    task = 'binary' if results['binary'] else 'ternary' # Get whether the classification task was ternary or binary.
    print('\nResults of training a', results['model_class'], f'classifier for {task} classification.')
    print('\tFeature type:', results['feature_type'])
    print('\tBalanced accuracy on training dataset:', results['training_acc'])
    print('\tBalanced accuracy on validation dataset:', results['validation_acc'])


def check_args(args):
    '''Check the command-line arguments.'''
    assert (not args.binary) or (args.model_class != 'nonlinear'), 'The Nonlinear model does not currently support binary classification.'
    assert args.out.split('.')[-1] == args.output_format, f'Output file type does not match specified format {args.output_format}.'


# TODO: Allow JSON output format.
# TODO: Add something to save model weights.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic'], default='nonlinear', help='The type of model to train.')
    parser.add_argument('--feature-type', '-f', type=str, default='KO', choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--out', '-o', default='results.pkl', help='The location to which the pickled results will be written.')
    parser.add_argument('--output-format', default='pkl', choices=['pkl', 'json'], help='Format of the results file.')
    parser.add_argument('--binary', '-b', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    
    args = parser.parse_args()
    check_args(args)

    # Load the cleaned-up training and validation datasets.
    training_dataset, validation_dataset = load_datasets(args.feature_type, binary=args.binary)
    # Unpack the numpy arrays stored in the dictionary. 
    X, y, X_val, y_val = training_dataset['features'], training_dataset['labels'], validation_dataset['features'], validation_dataset['labels']

    if getattr(args, 'model-class') == 'nonlinear':
        # TODO: Allow command-line specification of some key parameters, perhaps. Like the number of epochs.
        params = {'input_dim':X.shape[-1], 'n_epochs':400, 'lr':0.00001}
        model = train_nonlinear(X, y, X_val=X_val, y_val=y_val, params=params)
    if getattr(args, 'model-class') == 'logistic':
        params = dict()
        model = train_logistic(X, y)
    
    results = evaluate(model, X, y, X_val, y_val)
    # Store the model parameters in the results dictionary. 
    for param in params:
        results[param] = param
    results['feature_type'] = args.feature_type # Add feature type to the results.
    results['model_class'] = getattr(args, 'model-class')
    results['binary'] = args.binary

    print_summary(results) # Print a summary of the training run to the terminal. 

    print(f'\nWriting results to {args.out}.')
    if args.output_format == 'pkl': # If specified, save results in a pickle file.
        with open(args.out, 'wb') as f:
            pickle.dump(results, f) 
    elif args.output_format == 'json': # If specified, save results in a json file.
        with open(args.out, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)



# pretty_feature_names = {
#     'KO': 'all gene families',
#     'embedding.genome': 'genome embedding',
#     'embedding.geneset.oxygen': '5 gene set',
#     'metadata.number_of_genes': 'number of genes',
#     'metadata.oxygen_genes': 'O$_2$ gene count',
#     'metadata.pct_oxygen_genes': 'O$_2$ gene percent',
#     'aa_1mer': 'amino acid counts',
#     'aa_2mer': 'amino acid dimers',
#     'aa_3mer': 'amino acid trimers',
#     'chemical': 'chemical features',
#     'nt_1mer': 'nucleotide counts',
#     'nt_2mer': 'nucleotide dimers',
#     'nt_3mer': 'nucleotide trimers',
#     'nt_4mer': 'nucleotide 4-mers',
#     'nt_5mer': 'nucleotide 5-mers',
#     'cds_1mer': 'CDS nucleotide counts',
#     'cds_2mer': 'CDS nucleotide dimers',
#     'cds_3mer': 'CDS nucleotide trimers',
#     'cds_4mer': 'CDS nucleotide 4-mers',
#     'cds_5mer': 'CDS nucleotide 5-mers'}