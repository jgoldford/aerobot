'''Functions for reading and writing data from files, as well as some utilities for the command-line interface.'''
import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
import json
import argparse
import pickle

CWD, _ = os.path.split(os.path.abspath(__file__))
ASSET_PATH = os.path.join(CWD, 'assets')
RESULTS_PATH = os.path.join(CWD, '..', 'results')

FEATURE_TYPES = ['KO', 'embedding.genome', 'embedding.geneset.oxygen', 'metadata', 'chemical'] 
FEATURE_TYPES += [f'nt_{i}mer' for i in range(1, 6)]
FEATURE_TYPES += [f'cds_{i}mer' for i in range(1, 6)]
FEATURE_TYPES += [f'aa_{i}mer' for i in range(1, 4)]

# Some feature types are stored as metadata fields.
FEATURE_SUBTYPES = ['metadata.number_of_genes', 'metadata.oxygen_genes', 'metadata.pct_oxygen_genes']


# # NOTE: Where are these used?
# def load_ko2ec():
#     p = os.path.join(ASSET_PATH, 'mappings/keggOrthogroupsToECnumbers.07Feb2023.csv')
#     return pd.read_csv(p, index_col=0)


# def load_oxygen_kos():
#     p = os.path.join(ASSET_PATH, 'mappings/ko_groups.oxygenAssociated.07Feb2023')
#     return pd.read_csv(p, index_col=0)


class NumpyEncoder(json.JSONEncoder):
    '''Encoder for converting numpy data types into types which are JSON-serializable. Based
    on the tutorial here: https://medium.com/@ayush-thakur02/understanding-custom-encoders-and-decoders-in-pythons-json-module-1490d3d23cf7'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class NumpyDecoder(json.JSONDecoder):
    # TODO: This might be overkill, but for converting JSON stuff back to numpy objects.
    pass


def read_params(args:argparse.ArgumentParser, model_class:str='nonlinear') -> Dict:
    '''Read in model parameters from an ArgumentParser object.

    :param args: An ArgumentParser populated with parameters from the command-line.
    :param model_class: The type of model which the parameters will be passed into. One of 'logistic', 'nonlinear'.
    :return
    '''
    params = dict()
    # Determine which parameters to look for in the args, depending on the specified model class.
    if model_class == 'nonlinear': 
        param_options = ['weight_decay', 'n_epochs', 'hidden_dim', 'alpha', 'lr', 'batch_size', 'early_stopping']
        params.update({param:getattr(args, param) for param in param_options})
        # Need to specify the number of classes for Nonlinear, as this is the dimension of the output layer.
        params.update({'n_classes':3 if not args.binary else 2})
    elif model_class == 'logisic':
        param_options = ['C', 'penalty', 'max_iter']
        params.update({param:getattr(args, param) for param in param_options})
    # If not logistic or nonlinear, as in the case of randrel and meanrel, params is an empty dictionary.
    return params


def save_results_dict(results:Dict, path:str, fmt:str='json') -> NoReturn:
    '''Write a dictionary of results to the output path.

    :param results: A dictionary containing results from a model run, cross-validation, etc.
    :param path: The path to write the results to. 
    :param format: Either 'json' or 'pkl', specifies how to save the results.
    '''
    if fmt == 'pkl': # If specified, save results in a pickle file.
        with open(path, 'wb') as f:
            pickle.dump(results, f) 
    elif fmt == 'json': # If specified, save results in a json file.
        with open(path, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)


def load_results_dict(path:str) -> Dict:
    '''Read in a dictionary of results (e.g. from the evaluate function) stored either as a JSON or pickle file.

    :param path: The location of the file containing the results.
    :return: A dictionary containing the results. 
    '''
    fmt = path.split('.')[-1]
    assert fmt in ['pkl', 'json'], 'read_results_dict: File is not in a supported format. Must be either .pkl or .json.'
    if fmt == 'json':
        with open(path, 'r') as f:
            results = json.load(f)
    elif fmt == 'pkl':
        with open(path, 'rb') as f:
            results = pickle.load(f)
    return results


def save_hdf(datasets:Dict[str, pd.DataFrame], path:str)-> NoReturn:
    '''Save a dictionary of pandas DataFrames as an HD5 file at the specified output path.

    :param datasets: A dictionary where the keys are strings and the values are pandas DataFrames.
    :param path: The path where the file will be saved.
    '''
    with pd.HDFStore(path) as store:
        for key, value in datasets.items():
            store[key] = value


def load_hdf(path:str, feature_type:str) -> Dict[str, pd.DataFrame]:
    '''Load an HDF file storing either the training or validation data into a dictionary.

    :param path: The path to the HDF file.
    :param feature_type: The feature type to load, i.e. the key in the HDF file.
    :return: A dictionary mapping strings to pandas DataFrames. Dictionary keys should be 'feature', which
        maps to the feature DataFrame, and 'labels', which maps to the labels DataFrame.
    '''
    dataset = dict()
    # feature_type can be None, which is used when working with MeanRelative and RandRelative models. 
    dataset['features'] = None if feature_type is None else pd.read_hdf(path, key=feature_type)
    dataset['labels'] = pd.read_hdf(path, key='labels')
    return dataset



