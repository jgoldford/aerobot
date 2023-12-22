import argparse
import os
import csv
import sys
import glob
import pandas as pd
from aerobot.utls import count_kmers
from aerobot.io import ASSET_PATH
from sklearn.linear_model import LogisticRegression
from aerobot.models import GeneralClassifier


def process_directory(directory):
    """
    Process all files in the given directory and return the results.
    """
    files = glob.glob(directory + '/*')
    return files

def write_csv(output, df):
    """
    Write data to a CSV file or stdout using Pandas.
    """
    if output:
        df.to_csv(output, index=False)
    else:
        df.to_csv(sys.stdout, index=False)

def main():
    parser = argparse.ArgumentParser(description="Aerobot to predict oxygen requirement pheonotypes directly from genomic data.")
    parser.add_argument("input", help="Single fasta file per genome containing amino acid sequences. Input can be a file or a directory, either with .fa or .fa.gz extensions.")
    parser.add_argument("-o", "--output", help="Output CSV file name (optional)")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = process_directory(args.input)
    else:
        raise ValueError("The input is neither a file nor a directory")
    # load the model and vocab
    classifier = GeneralClassifier(model_class=LogisticRegression)
    classifier = classifier.load(filename=f"{ASSET_PATH}/trained_models/aa3mer/model.bin")
    vocab = pd.read_csv(f"{ASSET_PATH}/trained_models/aa3mer/vocab.txt",header=None)[0].tolist()
    feature_vocab = pd.DataFrame(vocab,columns=["Vocab"])

    # run featurizer
    kmers = count_kmers(input_files,3)
    # build feature matrix with correct vocab
    feature_matrix = feature_vocab.set_index("Vocab").join(kmers).fillna(0).loc[vocab]
    # run model
    predictions = classifier.predict(feature_matrix.T.values)

    # parse and save results
    FileNames = [x.split("/")[-1] for x in feature_matrix]
    FilePaths = list(feature_matrix)
    results_df = pd.DataFrame({"FileName":FileNames,"FilePath":FilePaths,"Oxygen_Phenotype_Prediction":predictions})

    # Write to CSV or stdout
    write_csv(args.output, results_df)

if __name__ == "__main__":
    main()
