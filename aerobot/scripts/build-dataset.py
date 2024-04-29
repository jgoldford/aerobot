'''Merge the Madin and Jablonska datasets, removing redundant genomes. since they overlap somewhat. Then split them into training 
and validation sets with no repeated species across the two sets. '''
import numpy as np
import pandas as pd
from aerobot.io import save_hdf, ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES
from aerobot.chemical import chemical_get_features
import os
import subprocess
import wget
from typing import NoReturn, Tuple, Dict
import tables
import warnings
# Ignore some annoying warnings triggered when saving HDF files.
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

# TODO: Put the key and filename maps into a CSV file. 
# TODO: Possibly add the code for reading and writing maps to the io.py file?


def load_data_madin(path:str=os.path.join(ASSET_PATH, 'data/madin/madin_data.h5'), feature_type:str='ko') -> Dict[str, pd.DataFrame]:
    '''Load the training data from Madin et. al. This data is stored in an H5 file, as it is too large to store in 
    separate CSVs. 

    :param path: The path to the HD5 file containing the training data.
    :param feature_type: The feature type to load.
    :return: A dictionary containing the feature data and corresponding labels.'''
    assert feature_type in FEATURE_TYPES, f'load_training_data: feature_type must be one of {FEATURE_TYPES}'
    
    output = dict()
    output['labels'] = pd.read_hdf(path, key='labels')
    if feature_type == 'chemical':
        kwargs = dict()
        for f in ['aa_1mer', 'cds_1mer', 'aa_1mer', 'metadata']:
            df = pd.read_hdf(path, key='nt_1mer')
            kwargs.update({f + '_df':df})
        features = chemical_get_features(**kwargs)
    else:
        features = pd.read_hdf(path, key=key_map[feature_type])
    if feature_type == 'metadata': # NOTE: This is ported over from the build_datasets script.
        # Calculate the percentage of oxygen genes for the combined dataset and and add it to the dataset
        features['pct_oxygen_genes'] = features['oxygen_genes'] / features['number_of_genes']

    output['features'] = features # Add the features to the output dictionary. 
    return output


def merge_datasets(training_dataset:Dict[str, pd.DataFrame], validation_dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    '''Merge the training and validation datasets, ensuring that the duplicate entries are taken care of by standardizing
    the index labels. Note that this function DOES NOT remove duplicate entries.

    :param training_dataset: A dictionary containing the training features and labels. 
    :param validation_dataset: A dictionary containing the validation features and labels. 
    :return: A dictionary containing the combined validation and training datasets.  
    '''
    def standardize_index(dataset:Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        '''Standardizes the indices the labels and features DataFrames in a dataset
        by removing the '.d' suffix, where d is a digit. This allows for comparison between datasets.

        :param dataset: The dataset to operate on.
        :return: The dataset with standardized indices.
        '''
        for key, data in dataset.items():
            data.index = [i.split('.')[0] for i in data.index]
            dataset[key] = data
        return dataset

    # Standardize the indices so that the datasets can be compared.
    training_dataset, validation_dataset = standardize_index(training_dataset), standardize_index(validation_dataset)
    # Unpack the features and labels dataframes from the datasets.
    training_features, training_labels = training_dataset['features'], training_dataset['labels'].drop(columns=['annotation_file', 'embedding_file'])
    validation_features, validation_labels = validation_dataset['features'], validation_dataset['labels'].drop(columns=['annotation_file', 'embedding_file'])

    # Combine the datasets, ensuring that the features columns which do not overlap are removed (with the join='inner')
    features = pd.concat([training_features, validation_features], axis=0, join='inner')
    labels = pd.concat([training_labels, validation_labels], axis=0, join='outer')

    return {'features':features, 'labels':labels}


def autofill_taxonomy(labels:pd.DataFrame) -> pd.DataFrame:

    # Redefine ranks from lowest-level to highest-level. 
    levels = ['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']

    # I noticed that no entries have no assigned species, but some do not have an assigned genus. 
    # Decided to autofill genus with the species string. I checked to make sure that every non-NaN genus is 
    # consistent with the genus in the species string, so this should be OK.
    assert np.all(~labels.Species.isnull()), 'autofill_taxonomy: Some entries have no assigned Species'
    labels['Genus'] = labels['Species'].apply(lambda s : s.split(' ')[0])
    
    # tax, n_autofilled = [], 0
    # for genus, df in labels.groupby('Genus', dropna=False):
    #     # genus_idxs = np.where(labels.Genus.values == g)[0]
    #     n_init_unclassified = df[levels].isnull().values.sum()
    #     df[levels] = df[levels].fillna(method='ffill')
    #     n_final_unclassified = df[levels].isnull().values.sum()
    #     n_autofilled += n_init_unclassified - n_final_unclassified
    #     tax.append(df)
    
    # print(f'\tautofill_taxonomy: Autofilled {n_autofilled} taxonomy entries.')
    # tax = pd.concat(tax, axis=0)
    # assert np.all(np.sort(tax.index.values) == np.sort(labels.index.values)), 'autofill_taxonomy: The indices in the taxonomy DataFrame do not match the labels DataFrame.'
    # labels, tax = labels.align(tax, join='inner', axis=0) # Align the indices.
    # labels = labels.combine_first(tax)

    return labels



def fill_missing_taxonomy(labels:pd.DataFrame) -> pd.DataFrame:
    '''Fill in missing taxonomy information from the GTDB taxonomy strings. This is necessary because
    different data sources have different taxonomy information populated. Note that every entry should have
    either a GTDB taxonomy string or filled-in taxonomy data.

    :param labels: The combined training and validation labels DataFrame.
    :return: The labels DataFrame with corrected taxonomy.
    '''
    levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. 

    tax = labels.gtdb_taxonomy.str.split(';', expand=True) # Split the GTDB taxonomy strings.
    tax = tax.apply(lambda x: x.str.split('__').str[1]) # Remove the g__, s__, etc. prefixes.
    tax.columns = levels # Label the taxonomy columns. 
    # Use the tax DataFrame to fill in missing taxonomy values in the labels DataFrame
    labels = labels.replace('no rank', np.nan).combine_first(tax)
    # Make sure everything at least has a class label. Maybe check for other taxonomies while we are at it.
    labels = autofill_taxonomy(labels)

    for level in levels[::-1]: # Make sure all taxonomy has been populated.
        n_unclassified = np.sum(labels[level].isnull())
        if n_unclassified > 0:
            print(f'\tfill_missing_taxonomy: {n_unclassified} entries have no assigned {level.lower()}.')
        # assert n_unclassified == 0, f'fill_missing_taxonomy: {n_unclassified} entries have no assigned {level.lower()}.'
    # labels[levels] = labels[levels].fillna('no rank') # Fill any NaNs with a "no rank" string for consistency.
    labels[levels] = labels[levels].fillna('no rank') # Fill in all remaining blank taxonomies with 'no rank'
    return labels


def load_jablonska_data(path:str=os.path.join(ASSET_PATH, 'data/jablonska/'), feature_type:str='KO') -> Dict[str, pd.DataFrame]:
    '''Load the data from Jablonska et. al.

    :param path: The path to the directory containing the validation data files.
    :param feature_type: The feature type to load.
    :return: A dictionary containing the feature data and corresponding labels.'''
    assert feature_type in FEATURE_TYPES, f'load_validation_data: feature_type must be one of {FEATURE_TYPES}'
    
    output = dict()
    labels = pd.read_csv(os.path.join(path, 'jablonska_labels.csv'), index_col=0).rename(columns={'Oder':'Order'}) # Fix a typo in one of the columns. 
    output['labels'] = labels # Add the DataFrame to the output. 

    if feature_type == 'chemical':
        kwargs = dict()
        for f in ['aa_1mer', 'cds_1mer', 'aa_1mer', 'metadata']:
            df = pd.read_csv(os.path.join(path, f'jablonska_{f}.csv'), index_col=0)
            kwargs.update({f + '_df':df})
        features = chemical_get_features(**kwargs)
    else:
        features = pd.read_csv(os.path.join(path, filename_map[feature_type]), index_col=0)
        # If the feature type is one of the following, fill 0 values with NaNs.
        if feature_type in ['KO', 'embedding.genome', 'embedding.geneset.oxygen']:
            features.fillna(0, inplace=True)
        if feature_type == 'metadata':
            features.set_index('genome', inplace=True)
            # Calculate the percentage of oxygen genes for the combined dataset and and add it to the dataset
            features['pct_oxygen_genes'] = features['oxygen_genes'] / features['number_of_genes']
        if feature_type.startswith('nt_'):
            # Remove text after the '.' character in the index for nt feature types.
            features.index = [i.split('.')[0] for i in features.index]

    output['features'] = features # Add the features to the output dictionary. 
    return output


def download_data(dir_path:str=os.path.join(ASSET_PATH, 'train/')) -> NoReturn:
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
    :return: A 3-tuple (new_df, duplicate_ids, removed_ids), where new_df is the DataFrame
        with duplicates removed, duplicate_ids are the IDs that were identical duplicates (one is retained),
        and removed_ids are the IDs that were removed entirely due to inconsistent data.
    '''
    duplicate_ids = data.index[data.index.duplicated()]
    ids_to_remove = []

    for id_ in duplicate_ids:
        duplicate_entries = data.loc[id_] # Get all entries in the DataFrame which match the ID.
        # NOTE: Keeping the first entry prefers keeping the entries from the training dataset, due to how they are concatenated.
        # This should ensure that GTDB taxonomy is preferred in the labels DataFrames.
        first_entry = duplicate_entries.iloc[0] # Get the first duplicate entry.
        # Check if the duplicate entries are consistent. If not, remove. 
        if not all(duplicate_entries == first_entry):
            ids_to_remove.append(id_)

    data = data.drop(ids_to_remove, axis=0) # Remove the inconsistent entries.
    duplicated = data.index.duplicated() # Recompute duplicated entries.
    duplicate_ids = data.index[duplicated].tolist() # Get the IDs of the duplicate entries. 

    return data[~duplicated].copy(), duplicate_ids, ids_to_remove


def training_validation_split(all_datasets:Dict[str, pd.DataFrame], random_seed:int=42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
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



if __name__ == '__main__':

    download_data() # Download training data from Google Cloud if it has not been already.

    all_datasets = dict()
    for feature_type in FEATURE_TYPES:
        print(f'Building datasets for {feature_type}...')
        # Load in the datasets.
        training_dataset = load_training_data(feature_type=feature_type)
        validation_dataset = load_validation_data(feature_type=feature_type)

        print(f'\tMerging datasets...')
        dataset = merge_datasets(training_dataset, validation_dataset)
        features, labels = dataset['features'], dataset['labels']

        # Fill in gaps in the taxonomy data using the GTDB taxonomy strings.
        dataset['labels'] = fill_missing_taxonomy(dataset['labels'])

        for key, data in dataset.items(): # key is "features" or "labels"
            data, duplicate_ids, removed_ids = remove_duplicates(data)
            print(f'\tFound {len(duplicate_ids)} duplicates in {key} across training and validation datasets.')
            if len(removed_ids) > 0:
                print(f'\tRemoved {len(removed_ids)} inconsistent entries in {key}.')
            dataset[key] = data

        # If there are already labels in the dictionary, check to make sure the new labels are equal.
        if 'labels' in all_datasets: # NOTE: There should be 3587 labels, 3480 with no duplicates
            l1, l2 = len(dataset['labels']), len(all_datasets['labels'])
            assert l1 == l2, f'Labels are expected to be the same length, found lengths {l1} and {l2}. Failed on feature type {feature_type}.'
            assert np.all(dataset['labels'].physiology.values == all_datasets['labels'].physiology.values), f'Labels are expected to be the same across datasets. Failed on feature type {feature_type}.'
        else: # Add labels to the dictionary if they aren't there already.
            all_datasets['labels'] = dataset['labels']
        
        all_datasets[feature_type] = dataset['features']

    training_datasets, validation_datasets = training_validation_split(all_datasets)

    print('Saving the datasets...')
    save_hdf(all_datasets, os.path.join(ASSET_PATH, 'updated_all_datasets.h5'))
    save_hdf(training_datasets, os.path.join(ASSET_PATH, 'updated_training_datasets.h5'))
    save_hdf(validation_datasets, os.path.join(ASSET_PATH, 'updated_validation_datasets.h5'))


# pretty_feature_names = {
#     'KO': 'all gene families',
#     'embedding.genome': 'genome embedding',
#     'metadata':None,
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

# filename_map = {'chemical':None,'KO': 'Jablonska_FS.KOCounts.07Feb2023.csv', 
#                 'embedding.genome': 'Jablonska_FS.WGE.07Feb2023.csv', 
#                 'embedding.geneset.oxygen': 'Jablonska_FS.OGSE.07Feb2023.csv',
#                 'metadata': 'Jablonska_FS.AF.07Feb2023.csv'}
# filename_map.update({f'aa_{i}mer':f'Jablonska_aa_{i}_mer.16Jul2023.csv' for i in range(1, 4)})
# filename_map.update({f'nt_{i}mer':f'Jablonska.nucletoide_{i}mers.19Jul2023.csv' for i in range(1, 6)})
# filename_map.update({f'cds_{i}mer':f'Jablonska_cds_{i}mer_features.csv' for i in range(1, 6)})

# # Create a dictionary mapping each feature type to a key in the HD5 file.
# key_map = {f:f for f in FEATURE_TYPES} # Most keys are the same as the feature type names.
# key_map.update({'embedding.genome':'WGE', 'embedding.geneset.oxygen':'OGSE', 'metadata':'AF'})

# df = {'feature_type':[], 'pretty_feature_name':[], 'hdf_key':[], 'filename':[]}
# for feature_type in FEATURE_SUBTYPES + FEATURE_TYPES:
#     df['feature_type'] += [feature_type] 
#     if feature_type in FEATURE_SUBTYPES:
#         df['hdf_key'] += [key_map['metadata']]
#         df['filename'] += [filename_map['metadata']]
#     else:
#         df['hdf_key'] += [key_map[feature_type]]
#         df['filename'] += [filename_map[feature_type]]
#     df['pretty_feature_name'] += [pretty_feature_names[feature_type]]

# df = pd.DataFrame(df).set_index('feature_type')
# df.to_csv('feature_type_metadata.csv')



