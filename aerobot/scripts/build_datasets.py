import numpy as np
import pandas as pd

from aerobot.io import load_training_data, load_validation_data, ASSET_PATH
from aerobot.io import download_training_data

from os import path

"""
Merge the Madin and Jablonska datasets since they overlap somewhat. 

Then split them into training and validation sets with no repeated
species across the two sets.

The sampled validation set is chosen by taking 20% of each 
phylogenetic family in the dataset.
"""

FEATURE_TYPES = ['KO', 'embedding.genome', 'embedding.geneset.oxygen',
                 'metadata', 
                 'aa_1mer', 'aa_2mer', 'aa_3mer', 'chemical',
                 'nt_1mer', 'nt_2mer', 'nt_3mer', 'nt_4mer', 'nt_5mer',
                 'cds_1mer', 'cds_2mer', 'cds_3mer', 'cds_4mer', 'cds_5mer']

def check_and_drop_duplicates(data):
    """Examines duplicated IDs. 
    
    If the data are identical, keeps the first. If not, keeps none of the dups. 

    Returns:
        A 3-tuple (new_df, duplicate_ids, removed_ids) where new_df is the dataframe
        with duplicates removed, duplicate_ids are the IDs that were duplicates (one is retained),
        and removed_ids are the IDs that were removed entirely due to inconsistent data.
    """
    dups = data.index.duplicated()
    dup_ids = data.index[dups]
    ids_to_remove = []

    for dup_id in dup_ids:
        instances = data.loc[dup_id]
        same_data = all(instances == instances.iloc[0])
        if not same_data:
            ids_to_remove.append(dup_id)

    data = data.drop(ids_to_remove)

    # keep first instance of remaining duplicates
    dups = data.index.duplicated()
    duplicate_ids = data.index[dups].tolist()
    return data[~dups].copy(), duplicate_ids, ids_to_remove


def normalize_index(data):
    """Normalizes the index of a dataframe.

    Removes the suffix ".d" from the index, if present. 
    Here 'd' connotes a digit. 
    """
    index_list = data.index.tolist()
    index_list = [i.split('.')[0] for i in index_list]
    data.index = index_list
    return data


if __name__ == '__main__':
    download_training_data()

    all_datasets = {}

    for ft in FEATURE_TYPES:
        print(f'Building datasets for {ft}...')
        training_data = load_training_data(ft)
        validation_data = load_validation_data(ft)

        training_features = training_data['features']
        training_labels = training_data['labels']
        validation_features = validation_data['features']
        validation_labels = validation_data['labels']

        # print the shapes of the data
        print(f'\tTraining features shape: {training_features.shape}')
        print(f'\tTraining labels shape: {training_labels.shape}')
        print(f'\tValidation features shape: {validation_features.shape}')
        print(f'\tValidation labels shape: {validation_labels.shape}')

        training_features, dup_feat_ids, removed_feat_ids = check_and_drop_duplicates(
            training_features)
        print(f'\tFound {len(dup_feat_ids)} duplicates in {ft} training features.')
        print(f'\tRemoved {len(removed_feat_ids)} inconsistent duplicates.')

        training_labels, dup_label_ids, removed_label_ids = check_and_drop_duplicates(
            training_labels)
        print(f'\tFound {len(dup_label_ids)} duplicates in {ft} training labels.')
        print(f'\tRemoved {len(removed_label_ids)} inconsistent duplicates.')

        # merge validation labels into validation features
        validation_data = validation_features.merge(validation_labels, how='inner',
                                                    left_index=True, right_index=True)

        # merge training and validation data, check if duplicates match
        print('\tMerging training and validation features -- will split again later...')
        concat_features = pd.concat([training_features, validation_features], axis=0)
        concat_features, duplicate_ids, removed_ids = check_and_drop_duplicates(
            concat_features)
        print(f'\tFound {len(duplicate_ids)} duplicates in {ft} concatenated features.')
        print(f'\tRemoved {len(removed_ids)} inconsistent duplicates.')
        concat_features = normalize_index(concat_features)

        print('\tMerging training and validation labels -- will split again later...')
        concat_labels = pd.concat([training_labels, validation_labels], axis=0)
        concat_labels, duplicate_ids, removed_ids = check_and_drop_duplicates(
            concat_labels)
        print(f'\tFound {len(duplicate_ids)} duplicates in {ft} concatenated labels.')
        print(f'\tRemoved {len(removed_ids)} inconsistent duplicates.')
        concat_labels = normalize_index(concat_labels)

        print(f'\tConcatenated features shape: {concat_features.shape}')
        print(f'\tConcatenated labels shape: {concat_labels.shape}')
        all_datasets[ft] = concat_features

        # Labels *should* be the same for all feature types -- TODO: check this
        all_datasets['labels'] = concat_labels

    # For the "metadata" key, we can now calculate the percentage of oxygen genes
    # and add it to the dataset
    metadata = all_datasets['metadata']
    metadata['pct_oxygen_genes'] = metadata['oxygen_genes'] / metadata['number_of_genes']

    # Clean up the labels, populate missing taxonomy.
    labels = all_datasets['labels'].rename(columns={'Oder': 'Order'})
    ncbi_tax = labels.ncbi_taxonomy.str.split(';', expand=True)
    ncbi_tax = ncbi_tax.apply(lambda x: x.str.split('__').str[1])
    ncbi_tax.columns = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    # set labels taxonomy when it is missing -- different source data have different data populated
    labels = labels.combine_first(ncbi_tax)
    all_datasets['labels'] = labels

    # Count the number of IDs in each taxonomic class
    ids_by_class = labels.groupby('Class').apply(lambda x: x.index.tolist()).to_dict()
    # There is both 'no rank' and empty string. Combine them under 'no rank'.
    ids_by_class['no rank'].extend(ids_by_class.pop(''))
    counts_by_class = {k: len(v) for k, v in ids_by_class.items()}
    print('Number of IDs in each taxonomic class:')
    for k, v in counts_by_class.items():
        print(f'\t{k}: {v}')

    # Check for duplicate IDs in each dataset
    for ft, dataset in all_datasets.items():
        print(f'Checking for duplicates in {ft}...')
        dataset, dup_ids, removed_ids = check_and_drop_duplicates(dataset)
        print(f'\tFound {len(dup_ids)} duplicates in {ft}.')
        print(f'\tRemoved {len(removed_ids)} inconsistent duplicates.')
        all_datasets[ft] = dataset

    # Randomly select 20% of each phylogenetic family -- 
    # a phylogenetically balanced train/test split at the family level
    print('Splitting concatenated dataset into training and validation sets...')
    np.random.seed(91)
    validation_ids = []
    for family, ids in ids_by_class.items():
        n = int(0.2 * len(ids))
        validation_ids.extend(np.random.choice(ids, n, replace=False))

    # Split the concatenated dataset back into training and validation sets
    training_datasets = {}
    validation_datasets = {}
    for ft, concat_dataset in all_datasets.items():
        training_datasets[ft] = concat_dataset[~concat_dataset.index.isin(validation_ids)]
        validation_datasets[ft] = concat_dataset[concat_dataset.index.isin(validation_ids)]

    print('Saving the full dataset for later inspection...')
    output_path_full = path.join(ASSET_PATH, 'updated_full_dataset.h5')
    with pd.HDFStore(output_path_full) as store:
        for key, value in all_datasets.items():
            store[key] = value

    print('Saving updated training dataset as an hdf5 file...')
    output_path_training = path.join(ASSET_PATH, 'updated_training_dataset.h5')
    with pd.HDFStore(output_path_training) as store:
        for key, value in training_datasets.items():
            store[key] = value

    print('Saving updated validation dataset as an hdf5 file...')
    output_path_validation = path.join(ASSET_PATH, 'updated_validation_dataset.h5')
    with pd.HDFStore(output_path_validation) as store:
        for key, value in validation_datasets.items():
            store[key] = value
    

