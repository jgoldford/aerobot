'''Code for phylogenetic cross-validation of classification models. This is to determine how robust the classifier
is to phylogenetic differences. The resulting output can be used to generate plots of the form of Figure 2C in  
https://www.biorxiv.org/content/10.1101/2024.03.22.586313v1.full.pdf'''
from aerobot.dataset import dataset_to_numpy, dataset_load_all, dataset_load_training_validation
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from aerobot.io import FEATURE_SUBTYPES, FEATURE_TYPES, save_results_dict, read_params
import argparse
from aerobot.models import evaluate, Nonlinear, GeneralClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np 
import pandas as pd

# Idea is to train on a bunch of different phylogenetic categories, and then test on a holdout set.
# This will show if a model is robust to phylogeny.

# Note that some of the taxonomy values are "no rank"... how should I handle this? Just throw them out when 
# doing the holdout sets? 

# Rather than leaving one group out, I think it makes sense to do the holdout as normal, but just ensuring no leakage on
# the specified phylogenetic level. 

# Should I include the baselines used in the paper?

# def print_taxonomy_info(dataset:Dict[str, pd.DataFrame], level:str='Class'):
def print_taxonomy_info(level:str='Class', feature_type:str='KO'):
    '''Print taxonomy information about the entries in the input dataset. Note that the number of unclassified entries
    printed by this function does not necessarily match the build_datasets.py output, as the counting performed by that script
    is done prior to de-duplication.
    
    :param level: The taxonomic level for which to display information.
    :param feature_type: The feature type for which to load the data. Results should be the same regardless of feature type.
    '''
    # Make sure the dataset has been loaded as a pandas DataFrame, not a numpy array. 
    # assert isinstance(dataset['labels'], pd.DataFrame), 'print_phylogenetic_info: There is no phylogenetic information in the dataset.'
    assert level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'], f'print_phylogenetic_info: Phylogenetic level {level} is invalid.'
    dataset = dataset_load_all(feature_type=feature_type, to_numpy=False)
    labels = dataset['labels']
    tax_info = labels[[level]] # This is the GTDB taxonomy for each entry.
    n_unclassified = np.sum(tax_info[level].str.match('no rank').values) # Get number of unclassified entries. 
    tax_info = tax_info[~tax_info[level].str.match('no rank')] # Remove anything labeled "no rank"
    sizes = []
    print(f'Summary of GTDB taxonomy for level {level.lower()}...')
    for taxon, info in tax_info.groupby(level):
        sizes.append(len(info))
        print('\t' + taxon, f'(n={len(info)})')

    mean = np.round(np.mean(sizes), 2)
    std = np.round(np.std(sizes), 2)
    print(f'\nMean number of members per {level.lower()}: {mean} +/- {std}')
    print(f'Smallest {level.lower()} size: {min(sizes)}')
    print(f'Largest {level.lower()} size: {max(sizes)}')
    print(f'Number of unclassified entries:', n_unclassified)


def phylogenetic_cross_validation(dataset:Dict[str, pd.DataFrame], n_splits:int=5, level:str='Class', model_class:str='nonlinear', params:Dict={}) -> Dict:
    '''Perform cross-validation using holdout sets partitioned according to the specified taxonomic level. For example, if the 
    specified level is 'Class', then the closest relative to any member of the holdout set will be an organism in the same phylum. If 
    the level is 'Family', then the closest relative to any member of the holdout set will be an organism in the same order... etc.
    
    :param dataset: A dictionary with two keys, 'features' and 'labels', each of which map to a pandas
        DataFrame containing the feature and label data, respectively.
    :param n_splits: The number of folds for K-fold cross-validation. This must be no fewer than the number of groups.
    '''
    group_kfold = GroupKFold()
    groups = dataset['labels'][level].values # Extract the taxonomy labels from the labels DataFrame.
    dataset = dataset_to_numpy(dataset) # Convert the dataset to numpy arrays after extracting the taxonomy labels.
    X, y = dataset['features'], dataset['labels'] # Extract the array and targets from the numpy dataset.

    # Filter out anything which does not have a taxonomic classification at the specified level.
    X = X[groups != 'no rank', :]
    y = y[groups != 'no rank']
    groups = groups[groups != 'no rank']

    scores = []
    for train_idxs, test_idxs in group_kfold.split(X, y, groups=groups):
        if model_class == 'nonlinear':
            params.update({'input_dim':X.shape[-1]}) # Make sure input dimensions are included. 
            model = GeneralClassifier(model_class=Nonlinear, params=params)
            model.fit(X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs])
        elif model_class == 'logistic':
            model = GeneralClassifier(model_class=LogisticRegression, params=params)
            model.fit(X[train_idxs], y[train_idxs])
    
        # Evaluate the trained model on the holdout set.
        results = evaluate(model, X[train_idxs], y[train_idxs], X_val=X[test_idxs], y_val=y[test_idxs])
        scores.append(results['f1_score']) # Store the F1 score.

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model-class', choices=['nonlinear', 'logistic'], help='The type of model to train.')
    parser.add_argument('--n-splits', default=5, type=int, help='The number of folds for K-fold cross validation.')
    parser.add_argument('--feature-type', type=str, default='KO', choices=FEATURE_SUBTYPES + FEATURE_TYPES, help='The feature type on which to train.')
    parser.add_argument('--out', '-o', default='phylo_bias_results.pkl', help='The location to which the pickled results will be written.')
    parser.add_argument('--output-format', default='pkl', choices=['pkl', 'json'], help='Format of the results file.')
    parser.add_argument('--binary', default=0, type=bool, help='Whether to train on the binary classification task. If False, then ternary classification is performed.')
    # Optional parameters for Nonlinear classifiers. 
    parser.add_argument('--n-epochs', default=1000, type=int, help='The maximum number of epochs to train the Nonlinear classifier.') 
    parser.add_argument('--lr', default=0.00001, type=float, help='The learning rate for training the Nonlinear classifier.') 
    parser.add_argument('--weight-decay', default=0.01, type=float, help='The L2 regularization penalty to be passed into the Adam optimizer of the Nonlinear classifier.') 
    parser.add_argument('--batch-size', default=20, type=int, help='The size of the batches for Nonlinear classifier training') 
    parser.add_argument('--alpha', default=10, type=int, help='The early stopping threshold for the Nonlinear classifier.') 
    parser.add_argument('--early-stopping', default=1, type=bool, help='Whether or not to use early stopping during Nonlinear classifier training.') 
    parser.add_argument('--hidden-dim', default=512, type=int, help='The number of nodes in the second linear layer of the Nonlinear classifier.')
    # Optional parameters for LogisticRegression classifiers.
    parser.add_argument('--C', default=100, type=float, help='Inverse of regularization strength for the LogisticRegression classifier' ) 
    parser.add_argument('--penalty', default='l2', type=str, help='The norm of the penalty term for the LogisticRegression classifier.') 
    parser.add_argument('--max-iter', default=10000, type=int, help='Maximum number of iterations for the LogisticRegression classifier.') 

    args = parser.parse_args()
    model_class = getattr(args, 'model-class') # Get the model class to run.
    params = read_params(args, model_class=model_class) # Read in model parameter specifications from the command line input.
    dataset, _ = dataset_load_training_validation(args.feature_type, binary=args.binary, to_numpy=False) # Load the training dataset without converting to numpy arrays (yet).
    
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'] # Define the taxonomic levels. Kingdom is ommitted.

    # Should probably report the standard deviation, mean, and standard error for each run.
    scores = {l:{'std':None, 'mean':None, 'err':None} for l in levels}
    for level in levels:
        print(f'Performing phylogeny-based cross-validation with {level.lower()}-level holdout set.')
        # Retrieve the F1 scores for the level. 
        level_scores = phylogenetic_cross_validation(dataset, level=level, model_class=model_class, n_splits=args.n_splits, params=params)
        level_scores[level]['std'] = np.std(scores)
        level_scores[level]['err'] = np.std(scores) / np.sqrt(len(scores))
        level_scores[level]['mean'] = np.mean(scores)

    results = {'scores':scores}
    # Add other relevant information to the results dictionary.
    results.update(params)
    results['feature_type'] = args.feature_type # Add feature type to the results.
    results['model_class'] = model_class
    results['binary'] = args.binary

    print(f'\nWriting results to {args.out}.')
    save_results_dict(results, args.out, fmt=args.output_format)
