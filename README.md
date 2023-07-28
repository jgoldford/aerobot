# AEROBOT
python classifier for predicting oxygen usage in prokaryotes

<img src="https://cdn.discordapp.com/attachments/977537752591659008/1072408230862520420/JoshG_Ukiyo-e_style_drawing_of_a_cute_microbe_wearing_a_gas_mas_d5523523-2df3-42bd-b1b3-67d18237f331.png" alt="AEROBOT" title="AEROBOT" width="600" height="400" class="center">

## Project goals and to-dos:
https://docs.google.com/document/d/1GFvE4NhJ0x5uTzFr89tgN11egyMZdcE1W_MYV_TU5fQ/edit

1. Build annotation files for aerobe, facultative and anaerobe: DONE
2. Annotate genomes with KEGG orthogroups (KO) for both Westoby and Jabłońska data sets: 
3. Embed genomes using the ProtT5-XL-UniRef50 protein langauge model (ESM?): 
4. Build ML models (Westoby train, Jabłońska test)


## JG Notes: 07-Feb-2023
added training data to google cloud bucket and download functions. To install, using the following code:

## Installation

In a conda or virtual environment, clone git repo and install using pip.

```sh
conda create -n aerobot python=3.8
conda activate aerobot
git clone https://github.com/jgoldford/aerobot.git
cd aerobot
pip install -e .
# install other packages
pip install pandas tables requests wget biopython
pip install -U scikit-learn
pip install ipykernel
# makes life easier when working with juptyer notebooks
python -m ipykernel install --user --name aerobot --display-name "Python 3.8 (aerobot)"
```

## Downloading training data
The training data is >100MB,so I put everything on a google cloud bucket. Open a jupyter notebook and using the following code block to download data

```python
from aerobot.io import download_training_data
download_training_data()
```

## Loading training data
I curated four types of training data: 1. KO counts, 2. The mean embedding for all proteins in the genome (WGE), 3. the mean embedding for all oxygen associated proteins in the genome (OGSE), and (4) kmer counts for either amino acids or genomes (aa_1mer, aa_2mer, nt_1mer and so on). To load a feature set ready for ML work, using the following function

```python
from aerobot.io import load_training_data
training_data = load_training_data(feature_type="embedding.genome")
training_data["features"] # genome x feature dataframe
training_data["labels"] # dataframe with metadata/phenotypes for each genome
```

I also wrote a simple function to process the training data into numpy arrays for ML work:

```python
from aerobot.utls import process_data
cleaned_data = process_data(training_data["features"], training_data["labels"]["physiology"], validation_data["features"], validation_data["labels"]["physiology"])
```

this function just makes sure all the feature columns overlap and match for both training and validation sets

## Model training and validation
I wrote a simple class instance to handle data normalization and classification pipelines.  We can also save and load pretrained models for downstream applications.  Below is the code you would use to train a logistic classifier model, and evaluate on the test set.

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

## Model saving and loading for future use
Now that you've trained the model, ya gotta save it! Here is how you would save and load models for later use
```python
model.save("logistic.Classifier.EmbeddingFeatures.joblib")

# load model into a new object
new_model = LogisticClassifier.load("logistic.Classifier.EmbeddingFeatures.joblib")
```

## Some more analysis
Suppose you want to look at which classes the model was good or bad at classifying. You'd want to analyze the full confusion matrix.  This is weird name, since you'll likely be less confused after looking at it. Here is how you'd compute this quickly using this package:

```python
C = model.confusion_matrix(cleaned_data["X_train"], cleaned_data["y_train"])
C= pd.DataFrame(C,index=model.classifier.classes_,columns=model.classifier.classes_)
C.apply(lambda x: x/x.sum(),axis=1)
```
