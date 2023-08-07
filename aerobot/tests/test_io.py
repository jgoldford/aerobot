import pandas as pd
import unittest

from aerobot.io import load_training_data, load_validation_data, FEATURE_TYPES
from parameterized import parameterized


class IOTests(unittest.TestCase):
    """Unit tests for the training and validation data representations.
    
    Checks that these data are internally consistent in their layout and ids. 
    """
    def _check_data(self, feature_name, data_type, data_dict):
        features = data_dict['features']
        labels = data_dict['labels']
        feature_ids = set(features.index)
        label_ids = set(labels.index)

        desc = '{0}, {1}'.format(feature_name, data_type)
        merged = pd.merge(features, labels, left_index=True, right_index=True)

        # Should have some data post-merge, i.e. index formats match
        self.assertGreaterEqual(merged.shape[0], 0, desc)

        # Always have more labels than features due to how the dataset is constructed.
        self.assertGreaterEqual(labels.index.size, features.index.size)
        
    def _training_feature_test(self, feature_name):
        training_data = load_training_data(feature_name)
        self._check_data(feature_name, 'training', training_data)

    def _validation_feature_test(self, feature_name):
        validation_data = load_validation_data(feature_name)
        self._check_data(feature_name, 'validation', validation_data)

    @parameterized.expand(FEATURE_TYPES)
    def test_validation_feature(self, feature_name):
        self._training_feature_test(feature_name)

    @parameterized.expand(FEATURE_TYPES)
    def test_validation_feature(self, feature_name):
        self._validation_feature_test(feature_name)

    def _train_validation_feature_test(self, feature_name):
        training_data = load_training_data(feature_name)
        validation_data = load_validation_data(feature_name)
        train_data, test_data = train_data.align(test_data, axis=1, join='inner')

    @parameterized.expand(FEATURE_TYPES)
    def test_train_validation_features(self, feature_name):
        self._train_validation_feature_test(feature_name)

if __name__ == '__main__':
    unittest.main()