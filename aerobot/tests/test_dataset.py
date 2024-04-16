'''Tests for the training and validation datasets generated by the build_datasets.py script.'''
import pandas as pd
import unittest
from aerobot.io import ASSET_PATH, FEATURE_TYPES, FEATURE_SUBTYPES
from aerobot.dataset import dataset_load_training_validation
from parameterized import parameterized
import numpy as np

# TODO: Add "sanity check" test cases for organisms which we KNOW should be facultative, aerobe, etc.

class DatasetTests(unittest.TestCase):
    '''Unit tests for the training and validation data representations.'''

    def _test_more_labels_than_features(self, features:pd.DataFrame=None, labels:pd.DataFrame=None):
        # There should always be more labels (or equal to) than features.
        self.assertGreaterEqual(len(labels), len(features))

    def _test_label_and_feature_indices_match(self, features:pd.DataFrame=None, labels:pd.DataFrame=None):
        # Merge the labels and features on the index.
        merged_features = features.merge(labels, left_index=True, right_index=True, how='left')
        self.assertTrue(len(merged_features) == len(features)) # Make sure there is one label for every feature.

    def _test_no_duplicate_entries(self, features:pd.DataFrame=None, labels:pd.DataFrame=None):
        self.assertTrue(features.index.is_unique)
        self.assertTrue(labels.index.is_unique)

    @parameterized.expand(FEATURE_TYPES + FEATURE_SUBTYPES)
    def test_correct_type_in_numpy_arrays(self, feature_type:str):
        # Making sure that the index or column values don't end up in the data used to train models. 
        # I doubt this is happening, but just want to confirm. 
  
        # Load in the training and testing datasets as numpy arrays.
        training_dataset, validation_dataset = dataset_load_training_validation(feature_type, to_numpy=True)
        # Make sure everything is actually a numpy array of floats (or strings, in the case of labels).
        for dataset in [training_dataset, validation_dataset]:
            features_dtype = dataset['features'].dtype
            labels_dtype = dataset['labels'].dtype
            self.assertTrue(features_dtype == np.float32, msg=f'Observed feature data type is {features_dtype}, expected np.float32.')
            self.assertTrue(labels_dtype == np.object_, msg=f'Observed labels data type is {labels_dtype}, expected {np.object_}.')

    @parameterized.expand(FEATURE_TYPES + FEATURE_SUBTYPES)
    def test_feature_type_datasets(self, feature_type:str):
        # Load in the training and validation sets as pandas DataFrames.
        training_dataset, validation_dataset = dataset_load_training_validation(feature_type, to_numpy=False)
        # Run a series of tests checking for consistency within each dataset.
        for dataset in [training_dataset, validation_dataset]:
            self._test_more_labels_than_features(**dataset)
            self._test_label_and_feature_indices_match(**dataset)
            self._test_no_duplicate_entries(**dataset)

    @parameterized.expand(FEATURE_TYPES + FEATURE_SUBTYPES)
    def test_training_and_validation_sets_are_disjoint(self, feature_type:str):
        # Load in the training and validation sets as pandas DataFrames
        training_dataset, validation_dataset = dataset_load_training_validation(feature_type, to_numpy=False)
        training_ids = training_dataset['labels'].index
        validation_ids = validation_dataset['labels'].index
        # Already testing for no duplicate entries, so assume elements in training_ids and validation_ids are already unique.
        all_ids = np.concatenate([training_ids, validation_ids]).ravel()
        # Make sure there are no IDs in the training and validation datasets.
        self.assertTrue(len(all_ids) == len(np.unique(all_ids)))

if __name__ == '__main__':
    unittest.main()