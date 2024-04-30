
'''Code for loading and processing the training and validation datasets created by the build_datasets script.'''
import pandas as pd
import numpy as np
import os
import subprocess as sb
import wget
from typing import Dict, NoReturn, Tuple
from aerobot.chemical import chemical_get_features
from aerobot.io import load_hdf, FEATURE_SUBTYPES, FEATURE_TYPES, ASSET_PATH
import json

def dataset_align(dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    '''Align the features and labels in a dataset, so that the indices match.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: The input datset with the indices in the features and labels DataFrames matched and aligned.
    '''
    features, labels = dataset['features'], dataset['labels'] # Unpack the stored DataFrames.
    n = len(features) # Get the original number of elements in the feature DataFrame for checking later. 
    features, labels  = features.align(labels, join='inner', axis=0) # Align the indices.

    # Make sure everything worked as expected.
    assert np.all(np.array(features.index) == np.array(labels.index)), 'dataset_align: Indices in training labels and data do not align.'
    assert len(features) == n, f'dataset_align: {n - len(features)} rows of data were lost during alignment.'

    return {'features':features, 'labels':labels} # Return the aligned dataset.


def dataset_to_numpy(dataset:Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    '''Convert the input dataset, which is a dictionary of pandas DataFrames, to a dictionary mapping
    'features' and 'labels' to numpy arrays. The 'labels' array contains only the values in the 
    physiology column.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: The input dataset with the DataFrames converted to numpy arrays. The the feature array is of 
        type is np.float32 and size (n, d) where n is the number of entries and d is the feature dimension.
        The labels array is one dimensional and of length n, and is of type np.object_.
    '''
    numpy_dataset = dict() # Create a new dataset.
    numpy_dataset['features'] = dataset['features'].values # .astype(np.float32)
    numpy_dataset['labels'] = dataset['labels'].physiology.values
    return numpy_dataset


def dataset_clean(dataset:Dict[str, pd.DataFrame], to_numpy:bool=True, binary:bool=False) -> Tuple[Dict]:
    '''Clean up the input dataset by (1) aligning the feature and label indices, (2) formatting the physiology labels
    for the downstream classification task, (3) removing columns in the feature DataFrame with NaNs, and (4) converting
    the DataFrames to numpy arrays, if specified.

    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :param to_numpy: Whether or not to convert the feature sets to numpy ndarrays for model compatibility.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :return: The cleaned-up dataset.
    '''
    # Select a label map for the binary or ternary classification task.
    ternary_label_map = {"Aerobe": "aerobe", "Facultative": "facultative", "Anaerobe": "anaerobe"}
    binary_label_map = {"Aerobe": "tolerant", "Facultative": "tolerant", "Anaerobe": "intolerant"}
    label_map = binary_label_map if binary else ternary_label_map
    
    dataset['labels'].physiology = dataset['labels'].physiology.replace(label_map) # Format the labels.

    if dataset['features'] is not None:
        dataset['features'] = dataset['features'].dropna(axis=1) # Drop columns which contain NaNs the dataset. 
        dataset = dataset_align(dataset) # Align the features and labels indices.

    return dataset_to_numpy(dataset) if to_numpy else dataset


def dataset_load(feature_type:str, path:str) -> Dict:
    '''Load a dataset for a particular feature type from the specified path.
    
    :param feature_type: Feature type to load from the HDF file. If None, genome IDs are used 
        as the feature type (for working with MeanRelative and RandRelative classifiers).
    :param path: Path to the HDF dataset to load. 
    :return: A dictionary with keys 'features' and 'labels' containing the feature data and metadata.
    '''
    subtype = None
    assert feature_type in FEATURE_TYPES + FEATURE_SUBTYPES + [None], f'dataset_load: Input feature type {feature_type} is invalid.'
    if feature_type is not None:
        # Special case if the feature_type is a "subtype", which is stored as a column in the metadata.
        if feature_type in FEATURE_SUBTYPES:
            feature_type, subtype = feature_type.split('.')
        
    dataset = load_hdf(path, feature_type=feature_type) # Read from the HDF file.
    if dataset['features'] is None:
        dataset['features'] = pd.DataFrame({'genome_id':dataset['labels'].index}, index=dataset['labels'].index)
    if subtype is not None: # If a feature subtype is given, extract the information from the metadata.
        dataset['features'] = dataset['features'][[subtype]]

    return dataset


def dataset_load_all(feature_type:str, binary:bool=False, to_numpy:bool=True, drop_x:bool=True) -> Dict:
    '''Load the full dataset for the specified feature type.

    :param feature_type: The feature type for which to load data.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :param to_numpy: Whether or not to convert the feature sets to numpy ndarrays for model compatibility.
    :param drop_x: Whether or not to drop the X amino acids when amino acid feature sets are being loaded.
    :return: A dictionary with the cleaned-up full dataset.
    '''
    dataset = dataset_load(feature_type, os.path.join(ASSET_PATH, 'updated_all_datasets.h5')) # Read in the dataset.
    dataset['features'] = dataset['features'][dataset_load_feature_order(feature_type, drop_x=drop_x)] # Ensure the column ordering is consistent. 
    dataset = dataset_clean(dataset, binary=binary, to_numpy=to_numpy) # Clean up the dataset.
    return dataset


def dataset_get_features(dataset:Dict[str, pd.DataFrame]) -> np.ndarray:
    '''Extract the names of the columns in the features DataFrame for a given dataset.

    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :return: A numpy array of features in the same order as the columns in the features DataFrame. 
    '''
    assert isinstance(dataset['features'], pd.DataFrame), 'dataset_get_features: Input dataset must contain DataFrames.'
    features = dataset['features'].columns
    return features.to_numpy()


def dataset_load_feature_order(feature_type:str, drop_x:bool=True) -> np.ndarray:
    '''Load the columns ordering for a particular feature type. This function returns columns ordered
    in the same way as the training dataset, which is used as a reference throughout the project.

    :param feature_type: The feature type for which to load data.
    :param drop_x: Whether or not to drop the X amino acids when amino acid feature sets are being loaded.
    :return: A numpy array of features, which are the columns of the features DataFrame for the input feature type. 
    '''
    dataset = dataset_load(feature_type, os.path.join(ASSET_PATH, 'updated_training_datasets.h5')) # Load the training dataset. 
    features = dataset_get_features(dataset)
    if 'aa_' in feature_type: # Remove all unknown amino acids from the feature set.
        features = np.array([f for f in features if 'X' not in f])
    return features


def dataset_load_training_validation(feature_type:str, binary:bool=False, to_numpy:bool=True, drop_x:bool=True) -> Tuple[Dict]:
    '''Load training and validation datasets for the specified feature type.

    :param feature_type: The feature type for which to load data.
    :param binary: Whether or not to use the binary training labels. If False, the ternary labels are used.
    :param to_numpy: Whether or not to convert the feature sets to numpy ndarrays for model compatibility.
    :param drop_x: Whether or not to drop the X amino acids when amino acid feature sets are being loaded.
    :return: A 2-tuple of dictionaries with the cleaned-up training and validation datasets as numpy arrays.
    '''
    training_dataset = dataset_load(feature_type, os.path.join(ASSET_PATH, 'updated_training_datasets.h5'))
    validation_dataset = dataset_load(feature_type, os.path.join(ASSET_PATH, 'updated_validation_datasets.h5'))

    # Make sure the columns in the training and validation datasets are aligned. 
    validation_dataset['features'] = validation_dataset['features'][dataset_load_feature_order(feature_type, drop_x=drop_x)]
    training_dataset['features'] = training_dataset['features'][dataset_load_feature_order(feature_type, drop_x=drop_x)]
    assert np.all(dataset_get_features(training_dataset) == dataset_get_features(validation_dataset)), 'dataset_load_training_validation: Column labels in training and validation datasets are not aligned.'

    # Clean up both datasets.
    validation_dataset = dataset_clean(validation_dataset, binary=binary, to_numpy=to_numpy)
    training_dataset = dataset_clean(training_dataset, binary=binary, to_numpy=to_numpy)

    return training_dataset, validation_dataset