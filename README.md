# AEROBOT
python classifier for predicting oxygen usage in prokaryotes

<img src="https://cdn.discordapp.com/attachments/977537752591659008/1072408230862520420/JoshG_Ukiyo-e_style_drawing_of_a_cute_microbe_wearing_a_gas_mas_d5523523-2df3-42bd-b1b3-67d18237f331.png" alt="AEROBOT" title="AEROBOT" width="600" height="400" class="center">

## Project goals and to-dos:
https://docs.google.com/document/d/1GFvE4NhJ0x5uTzFr89tgN11egyMZdcE1W_MYV_TU5fQ/edit

1. Build annotation files for aerobe, facultative and anaerobe: DONE
2. Annotate genomes with KEGG orthogroups (KO) for both Westoby and Jabłońska data sets: 
3. Embed genomes using the ProtT5-XL-UniRef50 protein langauge model (ESM?): 
4. Build ML models (Westoby train, Jabłońska test)

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
pip install pandas
pip install tables
pip install requests
pip install biopython
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
I curated three types of training data: 1. KO counts, 2. The mean embedding for all proteins in the genome (WGE), and 3. the mean embedding for all oxygen associated proteins in the genome (OGSE). To load a feature set ready for ML work, using the following function

```python
from aerobot.io import load_training_data
training_data = load_training_data(feature_type="embedding.genome")
training_data["features"] # genome x feature dataframe
training_data["labels"] # dataframe with metadata/phenotypes for each genome
```
