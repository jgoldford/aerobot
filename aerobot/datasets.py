
'''Code for loading and processing the training and validation datasets created by the build_datasets script.'''
import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
from aerobot.chemical import get_chemical_features
from aerobot.io import load_hdf, FEATURE_SUBTYPES, FEATURE_TYPES, ASSET_PATH
import json

def clean_datasets(training_dataset:Dict[str, pd.DataFrame], validation_dataset:Dict[str, pd.DataFrame], to_numpy:bool=True) -> Tuple[Dict]:

    training_features, training_labels = training_dataset['features'], training_dataset['labels']
    validation_features, validation_labels = validation_dataset['features'], validation_dataset['labels']

    # Drop columns which contain NaNs from each dataset.  
    training_features = training_features.dropna(axis=1)
    validation_features = validation_features.dropna(axis=1)
    # Make sure the entries and labels match.
    training_features, training_labels = training_features.align(training_labels, join='inner', axis=0)
    validation_features, validation_labels = validation_features.align(validation_labels, join='inner', axis=0)
    # Make sure the column ordering is the same in training and validation datasets.
    training_features, validation_features = training_features.align(validation_features, join='inner', axis=1)
 
    # Make sure everything worked.
    assert np.all(np.array(training_features.columns) == np.array(validation_features.columns)), 'Columns in training and validation set do not align.'
    assert np.all(np.array(training_features.index) == np.array(training_labels.index)), 'Indices in training labels and data do not align.'
    assert np.all(np.array(validation_features.index) == np.array(validation_labels.index)), 'Indices in training labels and data do not align.'

    if to_numpy: # If specified, convert the features and labels to numpy arrays. 
        # Explicitly converting to floats should catch any weirdness with index labels ending up in the result.
        training_features = training_features.values.astype(np.float32)
        validation_features = validation_features.values.astype(np.float32)
        # Extract the training and validation labels, converting them to numpy arrays. 
        training_labels = training_labels.physiology.values
        validation_labels = validation_labels.physiology.values

    return {'features':training_features, 'labels':training_labels}, {'features':validation_features, 'labels':validation_labels}


def load_datasets(feature_type:str, binary:bool=False, to_numpy:bool=True) -> Tuple[Dict]:
    '''Load training and testing datasets for the specified feature type.

    :param feature_type: The feature type for which to load data.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :param to_numpy: Whether or not to convert the feature sets to numpy ndarrays for compatibility.
    :return: A 2-tuple of dictionaries with the cleaned-up training and validation datasets as numpy arrays.
    '''
    assert feature_type in FEATURE_TYPES + FEATURE_SUBTYPES, f'load_data: Input feature must be one of: {FEATURE_TYPES + FEATURE_SUBTYPES}'
    
    subtype = None
    if feature_type in FEATURE_SUBTYPES:
        feature_type, subtype = feature_type.split('.')
    training_dataset = load_hdf(os.path.join(ASSET_PATH, 'updated_training_datasets.h5'), feature_type=feature_type)
    validation_dataset = load_hdf(os.path.join(ASSET_PATH, 'updated_validation_datasets.h5'), feature_type=feature_type)
    if subtype is not None: # If a feature subtype is given, extract the information from the metadata.
        training_dataset['features'] = training_dataset['features'][[subtype]]
        validation_dataset['features'] = validation_dataset['features'][[subtype]]

    # Select a label map for the binary or ternary classification task.
    ternary_label_map = {"Aerobe": "aerobe", "Facultative": "facultative", "Anaerobe": "anaerobe"}
    binary_label_map = {"Aerobe": "tolerant", "Facultative": "tolerant", "Anaerobe": "intolerant"}
    label_map = binary_label_map if binary else ternary_label_map
    # Format the labels for binary or ternary classification.
    training_dataset['labels'].physiology = training_dataset['labels'].physiology.replace(label_map)
    validation_dataset['labels'].physiology = validation_dataset['labels'].physiology.replace(label_map)

    return clean_datasets(training_dataset, validation_dataset, to_numpy=to_numpy)