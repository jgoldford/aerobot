## Downloading Genomic data for validation dataset
To fetch information for each genome in validation dataset, install the ncbi datasets cli:

```sh
conda install -c conda-forge ncbi-datasets-cli
```
Then run the cli to download genome sequences for each refseq ID. Note if you want rna or protein, just change "genome" to "rna,protein"

```sh
datasets download genome accession --inputfile refseq.txt --include genome
```

This will create a file called ncbi_datasets.zip
