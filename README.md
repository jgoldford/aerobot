# AEROBOT
Python classifier for predicting oxygen usage from bacterial and archaeal genomes.

<p align="center">
  <img src="logo.png" alt="AEROBOT" title="AEROBOT" width="400" height="400"/>
</p>


## Installation

In a conda or virtual environment, clone git repo and install using pip.

```sh
conda create -n aerobot python=3.8
conda activate aerobot
git clone https://github.com/jgoldford/aerobot.git
cd aerobot
pip install -e .
# install other packages
pip -r requirements.txt
```

## Running the CLI

If you'd like to use the tool immediately and have either a file (`*.fa` or `*.fa.gz`) or a directory containing these files.  Note that each file should be a collection of amino acid sequences from a single genome, as the script will predict a single phenotype per file.  You can either specific an output directory with the `-o` flag or let the script write the `stdout`.  To get started, we have an example fasta file for *E. coli* MG1655 in the following directory `aerobot/assets/example_data`.  Here is how you would run the script on this example file:

```sh
python aerobot.py aerobot/assets/example_data/eco.fa > ecoli_predictions.csv
```

If run correctly, the output file should contain the file name, the full path for the file, and a prediction of oxygen phsiology (`Aerobe`, `Anaerobe`, or `Facultative`).  For *E. coli* MG1655, the result should be `Facultative`.  

## Code for re-building the models
The following sections walk though the process of downloading and processing training and validation datasets, as well as performing training and running evaluation.


### Downloading training data
The training data is >100MB, so we put everything on a google cloud bucket. In a jupyter notebook, use the following code block to download data

```python
from aerobot.io import download_training_data
download_training_data()
```

### Loading training data
We curated four types of training data: 1. KO counts, 2. The mean embedding for all proteins in the genome (WGE), 3. the mean embedding for all oxygen associated proteins in the genome (OGSE), and (4) kmer counts for either amino acids or genomes (aa_1mer, aa_2mer, nt_1mer and so on). To load a feature set ready for ML work, use the following function

```python
from aerobot.io import load_training_data
training_data = load_training_data(feature_type="embedding.genome")
training_data["features"] # genome x feature dataframe
training_data["labels"] # dataframe with metadata/phenotypes for each genome
```

 Process the training data into numpy arrays for ML work:

```python
from aerobot.utls import process_data
cleaned_data = process_data(training_data["features"], training_data["labels"]["physiology"], validation_data["features"], validation_data["labels"]["physiology"])
```

This function just makes sure all the feature columns overlap and match for both training and validation sets.

### Model training and validation
We wrote a simple class instance to handle data normalization and classification pipelines.  You can also save and load pretrained models for downstream applications.  Below is the code you would use to train a logistic classifier model, and evaluate on the test set.

```python
from aerobot.models import LogisticClassifier
print("Fitting model...")
model = LogisticClassifier(max_iter=1000000,normalize=True)
model.fit(cleaned_data["X_train"], cleaned_data["y_train"])
# compute classification accuracy
accuracy = model.score(cleaned_data["X_train"], cleaned_data["y_train"])
# compute balanced accuracy
balanced_accuracy = model.balanced_accuracy(cleaned_data["X_train"], cleaned_data["y_train"])
#print accuracy and balanced accuracy
print("Accuracy: " + str(accuracy))
print("Balanced Accuracy: " + str(balanced_accuracy))


# compute accuracy and balanced accuracy on test set
test_accuracy = model.score(cleaned_data["X_test"], cleaned_data["y_test"])
test_balanced_accuracy = model.balanced_accuracy(cleaned_data["X_test"], cleaned_data["y_test"])
print("Test Accuracy: " + str(test_accuracy))
print("Test Balanced Accuracy: " + str(test_balanced_accuracy))
```

### Model saving and loading for future use
Now that you've trained the model, save it. Here is how you would save and load models for later use
```python
model.save("model.bin")

# load model into a new object
new_model = LogisticClassifier.load("model.bin"")
```

## Some more analysis
Suppose you want to look at which classes the model was good or bad at classifying. You'd want to analyze the full confusion matrix.  Here is how you'd compute this quickly using this package:

```python
C = model.confusion_matrix(cleaned_data["X_train"], cleaned_data["y_train"])
C= pd.DataFrame(C,index=model.classifier.classes_,columns=model.classifier.classes_)
C.apply(lambda x: x/x.sum(),axis=1)
```
