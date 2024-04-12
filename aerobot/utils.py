import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio import Entrez, SeqIO
from collections import defaultdict
import gzip
import time


def count_aa_kmers(fasta_files, k):
    """
    load fasta files and count kmers for amino acid sequences
    :param fasta_files: list of fasta files
    :param k: kmer length
    """
    # Initialize a dictionary to store dictionaries
    dict_list = {}

    # Loop through each fasta file
    for fasta_file in fasta_files:
        # Initialize a default dictionary to store kmer counts
        kmer_counts = defaultdict(int)

        # Parse the fasta file and iterate through each record
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(record.seq)

            # Iterate through the sequence to generate kmers
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i: i + k]

                # Add the kmer to the dictionary and increment its count
                kmer_counts[kmer] += 1

        # Add the dictionary to the dict_list with the fasta_file as key
        dict_list[fasta_file] = kmer_counts

    # Convert the dictionary of dictionaries to a pandas dataframe
    df = pd.DataFrame(dict_list).fillna(0)

    return df


def count_kmers(fasta_files, k):
    """
    load fasta files and count kmers for nucleotide or amino acid sequences
    :param fasta_files: list of fasta files
    :param k: kmer length
    """
    # Initialize a dictionary to store dictionaries
    dict_list = {}

    # Loop through each fasta file
    for fasta_file in fasta_files:
        # Initialize a default dictionary to store kmer counts
        kmer_counts = defaultdict(int)

        # Check if the file has a .gz extension
        if fasta_file.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        # Open the file accordingly
        with opener(fasta_file, 'rt') as handle:
            # Parse the fasta file and iterate through each record
            for record in SeqIO.parse(handle, "fasta"):
                sequence = str(record.seq)

                # Iterate through the sequence to generate kmers
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i: i + k]

                    # Add the kmer to the dictionary and increment its count
                    kmer_counts[kmer] += 1

        # Add the dictionary to the dict_list with the fasta_file as key
        dict_list[fasta_file] = kmer_counts

    # Convert the dictionary of dictionaries to a pandas dataframe
    df = pd.DataFrame(dict_list).fillna(0)

    return df


def run_pca(X,n_components=2,normalize=True):
	results = {}
	pca = PCA(n_components=n_components)
	if normalize:
		Xn = StandardScaler().fit_transform(X.values)
	else:
		Xn = X.values
	Xpca = pca.fit_transform(Xn)

	labels = ["PC{x} ({y}%)".format(x=x,y=round(y*100,2)) for x,y in list(zip(range(1,n_components+1),pca.explained_variance_ratio_))]
	pdf = pd.DataFrame(Xpca,index=X.index,columns =labels)
	results["pdf"] = pdf
	results["pca"] = pca
	return results



def download_genomes_from_assembly(refseq_ids, output_folder="downloaded_genomes", email="your_email@example.com"):
    Entrez.email = email

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for refseq_id in refseq_ids:
        # Fetch the assembly summary for the given RefSeq ID
        handle = Entrez.esummary(db="assembly", id=refseq_id)
        summary = Entrez.read(handle)
        handle.close()

        # Extract the nucleotide accession for the genomic sequence
        nuccore_id = summary['DocumentSummarySet']['DocumentSummary'][0]['AssemblyAccession']

        # Fetch the genomic sequence
        handle = Entrez.efetch(db="nucleotide", id=nuccore_id, rettype="fasta", retmode="text")
        filename = os.path.join(output_folder, f"{refseq_id}.fasta")
        
        with open(filename, 'w') as output_file:
            output_file.write(handle.read())
        handle.close()

        print(f"Downloaded {refseq_id} to {filename}")
        
        time.sleep(1)  # Respectful delay

