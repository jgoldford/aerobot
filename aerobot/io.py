'''Functions for reading and writing data from files.'''
import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
from aerobot.chemical import get_chemical_features
import json

CWD, _ = os.path.split(os.path.abspath(__file__))
ASSET_PATH = os.path.join(CWD, 'assets')

FEATURE_TYPES = ['KO', 'embedding.genome', 'embedding.geneset.oxygen', 'metadata', 'chemical'] 
FEATURE_TYPES += [f'nt_{i}mer' for i in range(1, 6)]
FEATURE_TYPES += [f'cds_{i}mer' for i in range(1, 6)]
FEATURE_TYPES += [f'aa_{i}mer' for i in range(1, 4)]

# Some feature types are stored as metadata fields.
FEATURE_SUBTYPES = ['metadata.number_of_genes', 'metadata.oxygen_genes', 'metadata.pct_oxygen_genes']


# NOTE: Where are these used?
def load_ko2ec():
    p = os.path.join(ASSET_PATH, 'mappings/keggOrthogroupsToECnumbers.07Feb2023.csv')
    return pd.read_csv(p, index_col=0)


def load_oxygen_kos():
    p = os.path.join(ASSET_PATH, 'mappings/ko_groups.oxygenAssociated.07Feb2023')
    return pd.read_csv(p, index_col=0)


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


def save_results_dict(results:Dict, path:str, format:str='json') -> NoReturn:
    '''Write a dictionary of results to the output path.

    :param results: A dictionary containing results from a model run, cross-validation, etc.
    :param path: The path to write the results to. 
    :param format: Either 'json' or 'pkl', specifies how to save the results.
    '''
    if format == 'pkl': # If specified, save results in a pickle file.
        with open(path, 'wb') as f:
            pickle.dump(results, f) 
    elif format == 'json': # If specified, save results in a json file.
        with open(args.out, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)


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
    output = dict()
    output['features'] = pd.read_hdf(path, key=feature_type)
    output['labels'] = pd.read_hdf(path, key='labels')
    return output



