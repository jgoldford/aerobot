'''A script for training a Nonlinear-based or Logistic-based GeneralClassifier.'''
import pandas as pd
import numpy as np
import tqdm
from aerobot.io import ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES, save_results_dict, read_params
from aerobot.dataset import dataset_load_feature_order
from aerobot.models import GeneralClassifier, evaluate, Nonlinear
from sklearn.linear_model import LogisticRegression
# from joblib import Parallel, delayed
from os import path
import argparse
from typing import Dict, NoReturn
import time
import pickle

def check_args(args:argparse.ArgumentParser):
    # TODO
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model-path', help='Path to the saved pre-trained model.')
    parser.add_argument('data-path', help='Path to the data on which to run the trained model. This should be in CSV format.')
    parser.add_argument('--feature-type', type=str, default='aa_3mer', choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--out', '-o', default='run_pretrained_results.csv', help='The location to which the predictions will be written.')


    args = parser.parse_args()
    check_args(args)
    t1 = time.perf_counter()

    data_path = getattr(args, 'data-path')
    data = pd.read_csv(data_path, index_col=0) # Need to preserve the index, which in the case of EMB data is the ID.
    feature_order = dataset_load_feature_order(args.feature_type)
    # Make sure all required features are present in the input data. 
    assert np.all(np.isin(feature_order, data.columns.to_numpy())), 'There are some required features missing in the input data.'
    data = data[feature_order] # Ensure consistent ordering of the columns. 

    model = GeneralClassifier.load(getattr(args, 'model-path')) # Load the trained model. 
    X = data.values # Extract the raw data from the input DataFrame.

    y_pred = model.predict(X)

    results = pd.DataFrame(index=data.index)
    results['prediction'] = y_pred.ravel() # Ravel because Nonlinear output is a column vector. 
    
    print(f'\nWriting results to {args.out}.')
    results.to_csv(args.out)
    
    t2 = time.perf_counter()
    print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

