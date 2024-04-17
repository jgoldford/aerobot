'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import numpy as np
import tqdm
from aerobot.io import ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES, save_results_dict 
from aerobot.dataset import dataset_load_training_validation
from aerobot.models import train_nonlinear, train_logistic, evaluate 
# from joblib import Parallel, delayed
from os import path
import argparse
from typing import Dict, NoReturn
import time
import pickle



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
    parser.add_argument('--out', '-o', default='run_model_results.pkl', help='The location to which the pickled results will be written.')
    parser.add_argument('--output-format', default='pkl', choices=['pkl', 'json'], help='Format of the results file.')
    parser.add_argument('--binary', '-b', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    
    args = parser.parse_args()
    check_args(args)
    t1 = time.perf_counter()

    # Load the cleaned-up training and validation datasets.
    training_dataset, validation_dataset = dataset_load_training_validation(args.feature_type, binary=args.binary, to_numpy=True)
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
    save_results_dict(results, args.out, format=args.output_format)
    
    t2 = time.perf_counter()
    print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

