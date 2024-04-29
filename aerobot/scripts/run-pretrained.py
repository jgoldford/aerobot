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
from warnings import simplefilter
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def check_args(args:argparse.ArgumentParser):
    # TODO
    pass

def clean_features(data:pd.DataFrame, feature_type:str='aa_3mer'):
    '''Make sure the features in the input data match the features (including order) of the data on 
    which the model was trained.

    :param data: The data on which to run the model.
    :param feature_type: The feature type of the data.
    '''
    feature_order = dataset_load_feature_order(feature_type) # Load in the correct features.
    
    missing = 0
    for f in feature_order:
        # If the data is missing a feature, fill it in with zeros.
        if f not in data.columns:
            missing += 1
            data[f] = np.zeros(len(data))

    print(missing, feature_type, 'features are missing from the input data. Filled missing data with 0.')
    data = data[feature_order] # Ensure the feature ordering is consistent. 
    return data

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
    data = pd.read_csv(data_path, index_col=0) # Need to preserve the index, which is the genome ID.
    data = clean_features(data, feature_type=args.feature_type) # Make sure the feature ordering is correct. 

    model = GeneralClassifier.load(getattr(args, 'model-path')) # Load the trained model. 
    X = data.values # Extract the raw data from the input DataFrame.
    y_pred = model.predict(X)

    results = pd.DataFrame(index=data.index) # Make sure to add the index back in!
    results['prediction'] = y_pred.ravel() # Ravel because Nonlinear output is a column vector. 
    
    print(f'\nWriting results to {args.out}.')
    results.to_csv(args.out)
    
    t2 = time.perf_counter()
    print(f'\nModel run complete in {np.round(t2 - t1, 2)} seconds.')

