'''Merge the Madin and Jablonska datasets, removing redundant genomes. since they overlap somewhat. Then split them into training 
and validation sets with no repeated species across the two sets. '''
import numpy as np
import pandas as pd
from aerobot.io import save_hdf, ASSET_PATH, FEATURE_TYPES
from aerobot.chemical import get_chemical_features
import os
import subprocess
import wget
from typing import NoReturn, Tuple, Dict
import tables
import warnings
# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


def load_training_data(path:str=os.path.join(ASSET_PATH, 'train/training_data.h5'), feature_type:str='KO') -> Dict[str, pd.DataFrame]:
    '''Load the training data for a specific feature from an HD5 file (specifically, the one downloaded)

    :param path: The path to the HD5 file containing the training data.
    :param feature_type: The feature type to load.
    :return: A dictionary containing the feature data and corresponding labels.'''
    output = dict()
    output['labels'] = pd.read_hdf(path, key='labels') # Extract the labels DataFrame from the HD5 file.
    # NOTE: This block of code was causing issues with the nt_ indices.
    # # If the feature type is from the nt class, keep only the string after the first '_' character in the labels DataFrame index.
    # if feature_type.startswith('nt_'):
    #     labels.index = ['_'.join(i.split('_')[1:]) for i in labels.index]

    # Create a dictionary mapping each feature type to a key in the HD5 file.
    key_map = {f:f for f in FEATURE_TYPES} # Most keys are the same as the feature type names.
    key_map.update({'embedding.genome':'WGE', 'embedding.geneset.oxygen':'OGSE', 'metadata':'AF'})

    assert feature_type in FEATURE_TYPES, f'load_training_data: feature_type must be one of {FEATURE_TYPES}'

    if feature_type == 'chemical':
        metadata_df = pd.read_hdf(path, key=key_map['metadata'])
        nt1_df = pd.read_hdf(path, key=key_map['nt_1mer'])
        aa1_df = pd.read_hdf(path, key=key_map['aa_1mer'])
        cds1_df = pd.read_hdf(path, key=key_map['cds_1mer'])
        features = get_chemical_features(metadata_df=metadata_df, aa1_df=aa1_df, nt1_df=nt1_df, cds1_df=cds1_df)
    else:
        features = pd.read_hdf(path, key=key_map[feature_type])
    if feature_type == 'metadata': # NOTE: This is ported over from the build_datasets script.
        # Calculate the percentage of oxygen genes for the combined dataset and and add it to the dataset
        features['pct_oxygen_genes'] = features['oxygen_genes'] / features['number_of_genes']

    output['features'] = features # Add the features to the output dictionary. 
    return output


def load_validation_data(path:str=os.path.join(ASSET_PATH, 'validation/features/'), feature_type:str='KO') -> Dict[str, pd.DataFrame]:
    '''Load the valudation data for a specific feature.

    :param path: The path to the directory containing the validation data files.
    :param feature_type: The feature type to load.
    :return: A dictionary containing the feature data and corresponding labels.'''
    output = dict()
    output['labels'] = pd.read_csv(os.path.join(ASSET_PATH, 'validation/labels/Jablonska_Labels.07Feb2023.csv'), index_col=0)

    # Dictionary mapping each feature type to the file with the relevant data.
    filename_map = {'KO': 'Jablonska_FS.KOCounts.07Feb2023.csv', 
                    'embedding.genome': 'Jablonska_FS.WGE.07Feb2023.csv', 
                    'embedding.geneset.oxygen': 'Jablonska_FS.OGSE.07Feb2023.csv',
                    'metadata': 'Jablonska_FS.AF.07Feb2023.csv'}
    filename_map.update({f'aa_{i}mer':f'Jablonska_aa_{i}_mer.16Jul2023.csv' for i in range(1, 4)})
    filename_map.update({f'nt_{i}mer':f'Jablonska.nucletoide_{i}mers.19Jul2023.csv' for i in range(1, 6)})
    filename_map.update({f'cds_{i}mer':f'Jablonska_cds_{i}mer_features.csv' for i in range(1, 6)})

    assert feature_type in FEATURE_TYPES, f'load_validation_data: feature_type must be one of {FEATURE_TYPES}'

    if feature_type == 'chemical':
        metadata_df = pd.read_csv(os.path.join(path, filename_map['metadata']), index_col=0)
        nt1_df = pd.read_csv(os.path.join(path, filename_map['nt_1mer']), index_col=0)
        aa1_df = pd.read_csv(os.path.join(path, filename_map['aa_1mer']), index_col=0)
        cds1_df = pd.read_csv(os.path.join(path, filename_map['cds_1mer']), index_col=0)
        features = get_chemical_features(metadata_df=metadata_df, aa1_df=aa1_df, nt1_df=nt1_df, cds1_df=cds1_df)
    else:
        features = pd.read_csv(os.path.join(path, filename_map[feature_type]), index_col=0)
        # If the feature type is one of the following, fill 0 values with NaNs.
        if feature_type in ['KO', 'embedding.genome', 'embedding.geneset.oxygen']:
            features.fillna(0, inplace=True)
        if feature_type == 'metadata':
            features.set_index('genome', inplace=True)
            # Calculate the percentage of oxygen genes for the combined dataset and and add it to the dataset
            features['pct_oxygen_genes'] = features['oxygen_genes'] / features['number_of_genes']
        # TODO: Is this supposed to be different than with the training data? And it's only done with the feature DataFrame?
        if feature_type.startswith('nt_'):
            # Remove text after the '.' character in the index for nt feature types.
            features.index = [i.split('.')[0] for i in features.index]

    output['features'] = features # Add the features to the output dictionary. 
    return output


def download_raw_training_data(dir_path:str=os.path.join(ASSET_PATH, 'train/')) -> NoReturn:
    '''Download the training data from Google Cloud.
    
    :param dir_path: The directory into which the training data will be downloaded.
    '''
    zip_filename = 'training_data.tar.gz' # TODO: Does this contain the training data only?
    zip_file_path = os.path.join(dir_path, zip_filename)
    h5_filename = os.path.join(dir_path, 'training_data.h5')
    # Check to make sure the data has not already been downloaded.
    if not os.path.exists(os.path.join(dir_path, zip_filename)):
        print('Downloading data from Google Cloud bucket...')
        url = f'https://storage.googleapis.com/microbe-data/aerobot/{zip_filename}'
        wget.download(url, dir_path)
        print('Download complete.')
    # Check to make sure the data has not already been extracted.
    if not os.path.exists(os.path.join(dir_path, h5_filename)):
        print('Extracting feature data...')
        subprocess.call(f'tar -xvf {os.path.join(dir_path, zip_filename)} --directory {dir_path}', shell=True)


def remove_duplicates(data:pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    '''Checks for duplicate IDs. If the data are identical, the first entry is kept. If not, 
    then both duplicates are dropped.

    :param data: A DataFrame containing all data.
    return: A 3-tuple (new_df, duplicate_ids, removed_ids), where new_df is the DataFrame
        with duplicates removed, duplicate_ids are the IDs that were identical duplicates (one is retained),
        and removed_ids are the IDs that were removed entirely due to inconsistent data.
    '''
    duplicate_ids = data.index[data.index.duplicated()]
    ids_to_remove = []

    for id_ in duplicate_ids:
        duplicate_entries = data.loc[id_] # Get all entries in the DataFrame which match the ID.
        first_entry = duplicate_entries.iloc[0] # Get the first duplicate entry.
        # Check if the duplicate entries are consistent. If not, remove. 
        if not all(duplicate_entries == first_entry):
            ids_to_remove.append(id_)

    data = data.drop(ids_to_remove, axis=0) # Remove the inconsistent entries.
    duplicated = data.index.duplicated() # Recompute duplicated entries.
    duplicate_ids = data.index[duplicated].tolist() # Get the IDs of the duplicate entries. 

    return data[~duplicated].copy(), duplicate_ids, ids_to_remove


def standardize_index(data:pd.DataFrame) -> pd.DataFrame:
    '''Normalizes the index of a DataFrame by removing the '.d'
    suffix, where d is a digit. This allows for comparison between datasets.

    :param data: The DataFrame to operate on.
    :return: The input DataFrame with a standardized index.
    '''
    data.index = [i.split('.')[0] for i in data.index]
    return data


def train_val_split(all_datasets:Dict[str, pd.DataFrame], random_seed:int=91) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    '''Split concatenated feature dataset into training and validation sets using phylogeny.
    
    :param all_datasets: A dictionary mapping each feature type to the corresponding dataset.
    :param random_seed: A random seed for reproducibility.
    :return: A 2-tuple of dictionaries. Each dictionary maps each feature type to a training (first tuple element) or validation
        (second tuple element) dataset.
    '''
    labels = all_datasets['labels'] # Get the labels out of the dictionary.
    # Group IDs by phylogenetic class. Convert to a dictionary mapping class to a list of indices.
    ids_by_class = labels.groupby('Class').apply(lambda x: x.index.tolist()).to_dict()

    # Now that the problem with the nt_1mer labels is fixed, there are no blank classes.
    # Some of the classes are 'no rank' or an empty string. Combine them under 'no rank'.
    # ids_by_class['no rank'].extend(ids_by_class.pop(''))

    counts_by_class = {k: len(v) for k, v in ids_by_class.items()}
    print('Number of IDs in each taxonomic class:')
    for k, v in counts_by_class.items():
        print(f'\t{k}: {v}')

    np.random.seed(random_seed) # For reproducibility. 
    validation_ids = []
    for class_, ids in ids_by_class.items():
        n = int(0.2 * len(ids)) # Get 20 percent of the indices from the class for the validation set.
        validation_ids.extend(np.random.choice(ids, n, replace=False))

    # Split the concatenated dataset back into training and validation sets
    training_datasets, validation_datasets = dict(), dict()
    for feature_type, dataset in all_datasets.items():
        training_datasets[feature_type] = dataset[~dataset.index.isin(validation_ids)]
        validation_datasets[feature_type] = dataset[dataset.index.isin(validation_ids)]

    return training_datasets, validation_datasets


# TODO: Should we just use GTDB taxonomy, instead of filling in existing?
def fill_missing_taxonomy(labels:pd.DataFrame) -> pd.DataFrame:
    '''Fill in missing taxonomy information from the GTDB (or NCBI?) taxonomy strings. This is necessary because
    different data sources have different taxonomy information populated.

    :param labels: The combined training and validation labels DataFrame.
    :return: The labels DataFrame with corrected taxonomy.
    '''
    # tax = labels.ncbi_taxonomy.str.split(';', expand=True)
    tax = labels.gtdb_taxonomy.str.split(';', expand=True)
    tax = tax.apply(lambda x: x.str.split('__').str[1])
    tax.columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    # Use the tax DataFrame to fill in missing taxonomy values in the labels DataFrame.
    # labels = labels.combine_first(tax)
    # Use all GTDB taxonomy instead of merging. 
    for col in tax.columns:
        labels[col] = tax[col]
    return labels


if __name__ == '__main__':

    download_raw_training_data() # Download training data from Google Cloud if it has not been already.

    all_datasets = dict()
    for feature_type in  FEATURE_TYPES:
        print(f'Building datasets for {feature_type}...')
        training_data = load_raw_training_data(feature_type=feature_type)
        validation_data = load_raw_validation_data(feature_type=feature_type)

        training_features, training_labels = training_data['features'], training_data['labels']
        validation_features, validation_labels = validation_data['features'], validation_data['labels']

        # Merge training and validation data, and remove any duplicates.
        print('\tMerging training and validation features...')
        features = pd.concat([training_features, validation_features], axis=0)
        features = standardize_index(features) # NOTE: I think this should be called before de-duplicating.
        features, duplicate_ids, removed_ids = remove_duplicates(features)
        print(f'\tFound {len(duplicate_ids)} duplicates in {feature_type} across training and validation features.')
        print(f'\tRemoved {len(removed_ids)} inconsistent duplicates.')

        print('\tMerging training and validation labels...')
        labels = pd.concat([training_labels, validation_labels], axis=0)
        labels = standardize_index(labels)
        labels, duplicate_ids, removed_ids = remove_duplicates(labels)
        print(f'\tFound {len(duplicate_ids)} duplicates in {feature_type} across training and validation labels.')
        print(f'\tRemoved {len(removed_ids)} inconsistent duplicates.')

        print(f'\tShape of merged features dataset: {features.shape}')
        print(f'\tShape of merged labels: {labels.shape}')
        all_datasets[feature_type] = features

        labels = labels.rename(columns={'Oder': 'Order'}) # Clean up some of the column headers.
        # If there are already labels in the dictionary, check to make sure the new labels are equal.
        if 'labels' in all_datasets: # NOTE: There should be 3587 labels, 3480 with no duplicates
            l1, l2 = len(labels), len(all_datasets['labels'])
            assert l1 == l2, f'Labels DataFrames are expected to be the same length, found lengths {l1} and {l2}. Failed on feature type {feature_type}.'
            assert np.all(labels.physiology.values == all_datasets['labels'].physiology.values), f'Labels are expected to be the same across datasets. Failed on feature type {feature_type}.'
            assert np.all(labels.physiology.values == all_datasets['labels'].physiology.values), f'Labels are expected to be the same across datasets. Failed on feature type {feature_type}.'
        else: # Add labels to the dictionary if they aren't there already.
            all_datasets['labels'] = labels

    all_datasets['labels'] = fill_missing_taxonomy(all_datasets['labels'])
    training_datasets, validation_datasets = train_val_split(all_datasets)

    print('Saving the datasets...')
    save_hdf(all_datasets, os.path.join(ASSET_PATH, 'updated_all_datasets.h5'))
    save_hdf(training_datasets, os.path.join(ASSET_PATH, 'updated_training_datasets.h5'))
    save_hdf(validation_datasets, os.path.join(ASSET_PATH, 'updated_validation_datasets.h5'))



