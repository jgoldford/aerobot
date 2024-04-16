'''Code for phylogenetic cross-validation of classification models. This is to determine how robust the classifier
is to phylogenetic differences.'''
from aerobot.dataset import dataset_load_full, dataset_to_numpy
from sklearn.model_selection import LeaveOneGroupOut
from aerobot.io import FEATURE_SUBTYPES, FEATURE_TYPES, dataset_load_all
import argparse
from aerobot.models import train_nonlinear, train_logistic, evaluate 
from tqdm import tqdm

# Idea is to train on a bunch of different phylogenetic categories, and then test on a holdout set.
# This will show if a model is robust to phylogeny. 

def phylogenetic_cross_validation(dataset:Dict[str, pd.DataFrame], min_holdout_size:int=50, level:str='Class', model_class:str='nonlinear'):
    ''''''

    logo = LeaveOneGroupOut()

    groups = dataset['labels'][level].values # Extract the taxonomy labels from the labels DataFrame.
    dataset = dataset_to_numpy(dataset) # Convert the dataset to numpy arrays after extracting the taxonomy labels.
    X, y = dataset['features'], dataset['labels'] # Extract the array and targets from the numpy dataset.

    val_accs = []

    for train_idxs, test_idxs in tqdm(logo.split(X, y, group=groups), desc='phylogenetic_cross_validation: Performing phylogeny-based cross-validation...'):
        group_name = groups[test_idxs[0]]
        # The size of the holdout group must be greater than the minimum holdout size.
        if len(test_idxs) < min_holdout_size:
            print(f'phylogenetic_cross_validation: Skipped holdout set for {level.lower()} {group_name}. Minimum size requirement was not met.')
            continue

        if model_class == 'nonlinear':
            params = {'input_dim':X.shape[-1], 'n_epochs':400, 'lr':0.00001}
            model = train_nonlinear(X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs], params=params)
        if model_class == 'logistic':
            params = dict()
            model = train_logistic(X[train_idxs], y[test_idxs])
        # Evaluate the trained model on the holdout set.
        results = evaluate(model, X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs])
        val_accs.append(results['validation_accs']) # Store the validation accuracy.

    return val_accs





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', '-l', choices=[])
    parser.add_argument('model-class', '-m', choices=['nonlinear', 'logistic'], default='nonlinear', help='The type of model to train.')
    parser.add_argument('--feature-type', '-f', type=str, default='KO', choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--out', '-o', default='results.pkl', help='The location to which the pickled results will be written.')
    parser.add_argument('--output-format', default='pkl', choices=['pkl', 'json'], help='Format of the results file.')
    parser.add_argument('--binary', '-b', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    
    args = parser.parse_args()

    dataset = dataset_load_full(feature_type)

