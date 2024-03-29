import pandas as pd
import tqdm

from aerobot.io import ASSET_PATH
from aerobot.utls import process_data
from sklearn.linear_model import LogisticRegression
from aerobot.models import GeneralClassifier
from joblib import Parallel, delayed
from os import path

# Full list of feature types 
FEATURE_TYPES = ['KO', 'embedding.genome', 'embedding.geneset.oxygen',
                 'metadata.number_of_genes', 'metadata.oxygen_genes', 'metadata.pct_oxygen_genes',
                 'aa_1mer', 'aa_2mer', 'aa_3mer', 'chemical',
                 'nt_1mer', 'nt_2mer', 'nt_3mer', 'nt_4mer', 'nt_5mer',
                 'cds_1mer', 'cds_2mer', 'cds_3mer', 'cds_4mer', 'cds_5mer']

# A minimal list would be 
# FEATURE_TYPES = ['KO', 'chemical', 'embedding.geneset.oxygen', 'aa_1mer', 'aa_3mer',] 

pretty_feature_names = {
    'KO': 'all gene families',
    'embedding.genome': 'genome embedding',
    'embedding.geneset.oxygen': '5 gene set',
    'metadata.number_of_genes': 'number of genes',
    'metadata.oxygen_genes': 'O$_2$ gene count',
    'metadata.pct_oxygen_genes': 'O$_2$ gene percent',
    'aa_1mer': 'amino acid counts',
    'aa_2mer': 'amino acid dimers',
    'aa_3mer': 'amino acid trimers',
    'chemical': 'chemical features',
    'nt_1mer': 'nucleotide counts',
    'nt_2mer': 'nucleotide dimers',
    'nt_3mer': 'nucleotide trimers',
    'nt_4mer': 'nucleotide 4-mers',
    'nt_5mer': 'nucleotide 5-mers',
    'cds_1mer': 'CDS nucleotide counts',
    'cds_2mer': 'CDS nucleotide dimers',
    'cds_3mer': 'CDS nucleotide trimers',
    'cds_4mer': 'CDS nucleotide 4-mers',
    'cds_5mer': 'CDS nucleotide 5-mers'
}

def load_training_data(feature_type):
    output = {'features': None, 'labels': None}
    feature_path = path.join(ASSET_PATH, "updated_training_dataset.h5")
    
    feature_matrix = pd.read_hdf(feature_path, key=feature_type)
    feature_matrix = feature_matrix.dropna(axis=1)

    label_matrix = pd.read_hdf(feature_path, key="labels")
    output["features"] = feature_matrix
    output["labels"] = label_matrix
    return output

def load_validation_data(feature_type):
    output = {'features': None, 'labels': None}
    feature_path = path.join(ASSET_PATH, "updated_validation_dataset.h5")
    
    feature_matrix = pd.read_hdf(feature_path, key=feature_type)
    feature_matrix = feature_matrix.dropna(axis=1)

    label_matrix = pd.read_hdf(feature_path, key="labels")
    output["features"] = feature_matrix
    output["labels"] = label_matrix
    return output


def get_model():
    params = {'penalty': 'l2', 'C': 100,
              'max_iter': 10000}
    model = GeneralClassifier(model_class=LogisticRegression,
                              params=params,
                              normalize=True)
    return model

ternary_label_map = {"Aerobe": "aerobe", "Facultative": "facultative", "Anaerobe": "anaerobe"}
binary_label_map = {"Aerobe": "tolerant", "Facultative": "tolerant", "Anaerobe": "intolerant"}

def get_data(feature_type, binary=False):
    feature_subtype = None
    if feature_type.startswith("metadata"):
        feature_type, feature_subtype = feature_type.split(".")

    training_data = load_training_data(feature_type=feature_type)
    validation_data = load_validation_data(feature_type=feature_type)

    if feature_subtype:
        training_data["features"] = training_data["features"][feature_subtype]
        validation_data["features"] = validation_data["features"][feature_subtype]

    label_map = ternary_label_map
    if binary:
        label_map = binary_label_map

    training_data["labels"]["physiology"] = training_data["labels"]["physiology"].replace(
        label_map)
    validation_data["labels"]["physiology"] = validation_data["labels"]["physiology"].replace(
        label_map)

    cleaned_data = process_data(training_data["features"],
                                training_data["labels"]["physiology"],
                                validation_data["features"],
                                validation_data["labels"]["physiology"])
    return training_data, validation_data, cleaned_data

def train_and_score(feature_type, binary=False):
    try:
        td, _, cleaned_data = get_data(feature_type, binary=binary)
        # if features are a series, get the name of the series
        training_features = td["features"]
        if isinstance(training_features, pd.Series):
            feature_names = [training_features.name]
        else:
            feature_names = training_features.columns

        print(f"training on {feature_type}")
        model = get_model()
        model.fit(cleaned_data["X_train"], cleaned_data["y_train"])
        accuracy = model.score(cleaned_data["X_train"], cleaned_data["y_train"])
        balanced_accuracy = model.balanced_accuracy(cleaned_data["X_train"], cleaned_data["y_train"])
        test_accuracy = model.score(cleaned_data["X_test"], cleaned_data["y_test"])
        test_balanced_accuracy = model.balanced_accuracy(cleaned_data["X_test"], cleaned_data["y_test"])

        # generate confusion matrix and flatten it to save
        confusion_matrix = model.confusion_matrix(cleaned_data["X_test"], cleaned_data["y_test"])
        # enumerate pairs of true and predicted labels
        confusion_matrix = pd.DataFrame(confusion_matrix, columns=model.classifier.classes_,
                                        index=model.classifier.classes_)

        n_iter = model.classifier.n_iter_[0]
        print(f"trained on {feature_type}")
        ret_dict = {
            "feature_set": feature_type,
            "feature_names": ','.join(feature_names),
            "pretty_feature_set": pretty_feature_names[feature_type],
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "test_accuracy": test_accuracy,
            "test_balanced_accuracy": test_balanced_accuracy,
            "model": model,
            "max_iter": model.classifier.max_iter,
            "n_iter": n_iter,
            "C": model.classifier.C,
            "converged": n_iter < model.classifier.max_iter
        }
        # for each pair of true and predicted labels, add a key 
        # to the dictionary with key true_label,predicted_label
        # and value count
        for true_label in confusion_matrix.index:
            for predicted_label in confusion_matrix.columns:
                ret_dict[f"{true_label},{predicted_label}"] = confusion_matrix.loc[true_label, predicted_label]
        return ret_dict

    except Exception as e:
        print(f"failed on {feature_type}")
        print(e)
        return None

# Assuming feature_types is a list of your feature types
results = Parallel(n_jobs=-1)(delayed(train_and_score)(ft) for ft in FEATURE_TYPES)
results = [r for r in results if r is not None]

accuracies = pd.DataFrame(results)
print('Ternary classification results')
print(accuracies)

# save accuracies to a pickle file
accuracies.to_pickle("logisticReg_l2_c100_1e4iter_models.ternary.pkl")

# Do it again for binary classification
results = Parallel(n_jobs=-1)(delayed(train_and_score)(ft, binary=True) for ft in FEATURE_TYPES)
results = [r for r in results if r is not None]

accuracies = pd.DataFrame(results)
print('Binary classification results')
print(accuracies)

# save accuracies to a pickle file
accuracies.to_pickle("logisticReg_l2_c100_1e4iter_models.binary.pkl")
