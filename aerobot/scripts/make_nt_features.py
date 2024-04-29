

import glob
import pandas as pd
from aerobot.io import asset_path
from aerobot.utls import count_kmers
import os

# Define the root directory
root_dir = f'{asset_path}/validation/genomes/ncbi_dataset/data/'

# Get a list of all .faa files under the root directory
file_paths = glob.glob(root_dir + '**/*.fna', recursive=True)

# Extract the genome ids from the file paths
genome_ids = [os.path.basename(os.path.dirname(path)) for path in file_paths]

# Convert the data into a pandas dataframe
validation_genomes = pd.DataFrame({
    'Genome_ID': genome_ids,
    'File_Path': file_paths
})

genome_fna_files = validation_genomes.File_Path.tolist()
file_mapper = validation_genomes.set_index("File_Path")["Genome_ID"].to_dict()

kmers = [1,2,3,4,5]
for k in kmers:
    df = count_kmers(genome_fna_files,k)
    df = df.T
    df.index = [file_mapper[x] for x in df.index.tolist()]
    df.to_csv(f"{asset_path}/validation/features/Jablonska.nucletoide_{k}mers.19Jul2023.csv")
    print(f"done with kmer: {k}")