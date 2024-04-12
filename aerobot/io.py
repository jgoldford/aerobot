'''Functions for reading and writing training and validation data.'''
import pandas as pd
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
from aerobot.utils import get_chemical_features

# NOTE: Can't pip install requirements on Python 3.12, needed 3.11.

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
    output['labels'] = pd.read_hdf(feature_path, key='labels')
    return output


def clean_datasets(training_dataset:Dict[str, pd.DataFrame], validation_dataset:Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray]]:

    training_features, training_labels = training_dataset['features'], training_dataset['labels']
    validation_features, validation_labels = validation_dataset['features'], validation_dataset['labels']

    # Drop columns which contain NaNs from each dataset.  
    training_features = training_features.dropna(axis=1)
    validation_features = validation_features.dropna(axis=1)
    # Make sure the entries and labels match.
    trianing_features, training_labels = training_features.align(training_labels, join='inner', axis=0)
    validation_features, validation_labels = validation_features.align(validation_labels, join='inner', axis=0)
    # Make sure the column ordering is the same in training and validation datasets.
    training_features, validation_features = training_features.align(validation_features, join='inner', axis=1)
    
    # Extract the training and validation labels, converting them to numpy arrays. 
    training_labels = training_labels.physiology.values
    validation_labels = validation_labels.physiology.values

    # Make sure everything worked.
    assert np.all(np.array(train_df.columns) == np.array(val_df.columns)), 'Columns in training and validation set do not align.'
    assert np.all(np.array(train_df.index) == np.array(train_labels_df.index)), 'Indices in training labels and data do not align.'
    assert np.all(np.array(val_df.index) == np.array(val_labels_df.index)), 'Indices in training labels and data do not align.'

    return {'features':trainig_features, 'labels':training_labels}, {'features':validation_features, 'labels':validation_labels}


def load_datasets(feature_type:str, binary:bool=False) -> Tuple[Dict[str, np.ndarray]]:
    '''Load training and testing datasets for the specified feature type.

    :param feature_type: The feature type for which to load data.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :return: A 2-tuple of dictionaries with the cleaned-up training and validation datasets as numpy arrays.
    '''
    assert feature_type in FEATURE_TYPES + FEATURE_SUBTYPES, f'load_data: Input feature must be one of: {FEATURE_TYPES + FEATURE_SUBTYPES}'
    
    subtype = None
    if feature_type.startswith('metadata'):
        feature_type, subtype = feature_type.split('.')
    training_dataset = load_hdf(os.path.join(ASSET_PATH, 'updated_training_datasets.h5'), feature_type=feature_type)
    validation_dataset = load_hdf(os.path.join(ASSET_PATH, 'updated_validation_datasets.h5'), feature_type=feature_type)
    if feature_subtype: # If a feature subtype is given, extract the information from the metadata.
        training_data['features'] = training_data['features'][subtype]
        validation_data['features'] = validation_data['features'][subtype]

    # Select a label map for the binary or ternary classification task.
    ternary_label_map = {"Aerobe": "aerobe", "Facultative": "facultative", "Anaerobe": "anaerobe"}
    binary_label_map = {"Aerobe": "tolerant", "Facultative": "tolerant", "Anaerobe": "intolerant"}
    label_map = binary_label_map if binary else ternary_label_map
    # Format the labels for binary or ternary classification.
    training_dataset['labels'].physiology = training_dataset['labels'].physiology.replace(label_map)
    validation_dataset['labels'].physiology = validation_dataset['labels'].physiology.replace(label_map)

    return clean_datasets(training_dataset, validation_dataset)


