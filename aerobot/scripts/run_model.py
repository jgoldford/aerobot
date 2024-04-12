'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import tqdm
from aerobot.io import ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES, load_datasets
from aerobot.utils import process_data
from sklearn.linear_model import LogisticRegression
from aerobot.models import GeneralClassifier, Nonlinear
# from joblib import Parallel, delayed
from os import path
import argparse
from typing import Dict
import pickle
import json


def train_logistic(X:np.ndarray, y:np.ndarray, params:Dict={'penalty':'l2', 'C':100, 'max_iter':10000}) -> GeneralClassifier:
    '''Train a LogisticRegression-based classifier.
    
    '''
    model = GeneralClassifier(model_class=LogisticRegression, params=params) # Instantiate a classifier.
    model.fit(X, y)
    return model


def train_nonlinear(X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray, params:Dict=None) -> GeneralClassifier:
    '''Train a Nonlinear-based classifier.
    
    '''
    model = GeneralClassifier(model_class=Nonlinear, params=params) # Instantiate a classifier.
    model.fit(X, y, X_val=X_val, y_val=y_val)
    return model


def evaluate(model:GeneralClassifier, X:np.ndarray, y:np.ndarray, X_val:np.ndarray, y_val:np.ndarray) -> Dict:
    '''Evaluate a trained GeneralClassifier using the training and test data.

    '''
    results = dict()

    training_acc = model.balanced_accuracy(X, y)
    validation_acc = model.balanced_accuracy(X_val, y_val)
    confusion_matrix = model.confusion_matrix(X_val, y_val)
    # confusion_matrix = pd.DataFrame(confusion_matrix, columns=model.classifier.classes_, index=model.classifier.classes_)
    tn, fp, fn, tp = confusion_matrix.ravel()
    results['tn'], results['fp'], results['fn'], results['tp'] = tn, fp, fn, tp
    results['classes'] = model.classifier.classes_

    results['feature_type'] = feature_type
    # results['pretty_feature_name'] = pretty_feature_names[feature_type]
    # results['confusion_matrix'] = confusion_matrix
    if hasattr(model, 'n_iter_'): # Only applies if the model used LogisticRegression.
        n_iter = model.classifier.n_iter_[0]
        results['n_iter'] = n_iter
        results['converged'] = n_iter < model.classifier.max_iter
    
    return results


def check_args(args):
    '''Check the command-line arguments.'''
    assert (not binary) or (args.model_class != 'nonlinear'), 'The Nonlinear model does not currently support binary classification.'
    assert args.out.split('.')[-1] == args.output_format, f'Output file type does not match specified format {args.output_format}.'


# TODO: Allow JSON output format.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic'], help='The type of model to train.')
    parser.add_argument('--feature-type', '-f', type=str, choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--out', '-o', default='results.pkl', help='The location to which the pickled results will be written.')
    parser.add_argument('--output-format', default='pkl', choices=['pkl', 'json'], help='Format of the results file.')
    parser.add_argument('--binary', '-b', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    
    args = parser.parse_args()
    check_args(args)

    # Load the cleaned-up training and validation datasets.
    training_dataset, validation_dataset = load_datasets(args.feature_type, binary=args.binary)
    # Unpack the numpy arrays stored in the dictionary. 
    X, y, X_val, y_val = training_dataset['features'], training_dataset['labels'], validation_dataset['features'], validation_dataset['labels']

    if args.model_class == 'nonlinear':
        # TODO: Allow command-line specification of some key parameters, perhaps. Like the number of epochs.
        params = dict()
        model = train_nonlinear(X, y, X_val, y_val)
    if args.model_class == 'logistic':
        params = dict()
        model = train_logistic(X, y)
    
    results = evalute(model, X, y, X_val, y_val)
    # Store the model parameters in the results dictionary. 
    for param in params:
        results[param] = params

    if args.output_format == 'pkl': # If specified, save results in a pickle file.
        with open(args.out, 'wb') as f:
            pickle.dump(results, f) 
    elif args.output_format == 'json': # If specified, save results in a json file.
        with open(args.out, 'w') as f:
            json.dump(results, fp)



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