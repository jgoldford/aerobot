import pandas as pd
#import h5py
import os
import subprocess as sb
import wget

from os import path

_cwd, _ = path.split(path.abspath(__file__))
ASSET_PATH = path.join(_cwd, 'assets')


FEATURE_TYPES = ['KO', 'embedding.genome', 'embedding.geneset.oxygen', 'metadata', 
                 'aa_1mer', 'aa_2mer', 'aa_3mer', 'nt_1mer', 'nt_2mer', 'nt_3mer',
                 'nt_4mer', 'nt_5mer', 'cds_1mer', 'cds_2mer', 'cds_3mer', 'cds_4mer',
                 'cds_5mer']


def download_training_data():
    # donwload the training data from google cloud

    fileName = "training_data.tar.gz"
    feature_path = path.join(ASSET_PATH, "train/")
    out_file_path = path.join(feature_path, fileName)
    train_h5_fname = path.join(feature_path, "training_data.h5")
    
    if not os.path.exists(out_file_path):
        print("downloading data from google cloud bucket...")
        url = "https://storage.googleapis.com/microbe-data/aerobot/{fn}".format(fn=fileName)
        wget.download(url, feature_path)
        print("")  # newline 
        print("download complete.")
    if not os.path.exists(train_h5_fname):
        print("extracting feature matrix..")
        sb.call("tar -xvf {fpath}{n} --directory {fpath}".format(fpath=feature_path,n=fileName),shell=True)
    print("done!")


CANONICAL_NTS = ['A', 'C', 'G', 'T']
GC_NTS = ['G', 'C']


def calc_gc(nt1_features_df):
    """Calculate the GC content of the canonical nucleotides.

    Args:
        nt1_features_df (pd.DataFrame): A dataframe containing the counts of
            single nucleotides.

    Returns:
        pd.Series: A series containing the GC content of each genome.
    """
    canonical_data = nt1_features_df[CANONICAL_NTS]
    gc = canonical_data[GC_NTS].sum(axis=1) / canonical_data.sum(axis=1)
    gc.name = 'gc_content'
    return gc


AA_NOSC_DF = pd.read_csv('aa_nosc.csv', index_col=0)
AA_NC = AA_NOSC_DF.NC
AA_ZC = AA_NOSC_DF.NOSC
CANONICAL_AAS = AA_NOSC_DF.index.tolist()


def calc_aa_features(aa1_features_df):
    """Calculate chemical features of the coding sequences.

    Calculates the formal C oxidation state of mRNAs, as well as 
    the C, N, O and S content. 

    Args:
        aa1_features_df (pd.DataFrame): A dataframe containing the counts of
            single amino acids.

    Returns:
        pd.Series: The formal C oxidation state.
    """
    aas = sorted(set(aa1_features_df.columns).intersection(CANONICAL_AAS))
    canonical_data = aa1_features_df[aas]
    NC_total = (canonical_data @ AA_NC[aas])
    Ne_total = (canonical_data @ (AA_ZC[aas] * AA_NC[aas]))
    mean_cds_zc = Ne_total / NC_total
    mean_cds_zc.name = 'cds_aa_zc'

    cols = [mean_cds_zc]
    for elt in 'CNOS':
        col = 'N{0}'.format(elt)
        ns = AA_NOSC_DF[col]
        totals = (canonical_data @ ns[aas])
        means = totals / canonical_data.sum(axis=1)
        means.name = 'cds_aa_{0}'.format(col)
        cols.append(means)
    return pd.concat(cols, axis=1)


NT_NOSC_DF = pd.read_csv('nt_nosc.csv')
RNA_NOSC_DF = NT_NOSC_DF[NT_NOSC_DF.type == 'RNA']
RNA_NOSC_DF = RNA_NOSC_DF.set_index('letter_code')
RNA_NC = RNA_NOSC_DF.NC
RNA_ZC = RNA_NOSC_DF.NOSC
# Includes DNA and RNA names
CANONICAL_NTS_ALL = RNA_NOSC_DF.index.unique().tolist()

def calc_rna_features(nt1_features_df):
    """Calculate chemical features of the RNA coding sequences.
    
    Calculates the formal C oxidation state of mRNAs, as well as 
    the C, N, O and S content. 

    Args:
        nt1_features_df (pd.DataFrame): A dataframe containing the counts of
            single nucleotides, either RNA or DNA. Assumed to single stranded.

    Returns:
        pd.Series: The formal C oxidation state of the RNA coding sequences.
    """
    my_nts = sorted(set(nt1_features_df.columns).intersection(CANONICAL_NTS_ALL))
    canonical_data = nt1_features_df[my_nts]
    NC_total = (canonical_data @ RNA_NC[my_nts])
    Ne_total = (canonical_data @ (RNA_ZC[my_nts] * RNA_NC[my_nts]))
    mean_cds_zc = Ne_total / NC_total
    mean_cds_zc.name = 'cds_nt_zc'

    cols = [mean_cds_zc]
    for elt in 'CNO':
        col = 'N{0}'.format(elt)
        ns = RNA_NOSC_DF[col]
        totals = (canonical_data @ ns[my_nts])
        means = totals / canonical_data.sum(axis=1)
        means.name = 'cds_nt_{0}'.format(col)
        cols.append(means)
    return pd.concat(cols, axis=1)


def load_training_data(feature_type="KO"):
    output = {"features":[],"labels":[]}
    feature_path = path.join(ASSET_PATH, "train/training_data.h5")
    
    #labels = pd.read_csv(asset_path+"/train/labels/Westoby.Trimmed.07Feb2023.csv")
    labels = pd.read_hdf(feature_path,key="labels")
    output["labels"] = labels
    
    key_dict = {
        "KO": "KO",
        "embedding.genome": "WGE",
        "embedding.geneset.oxygen": "OGSE",
        "metadata": "AF",
        "aa_1mer": "aa_1mer",
        "aa_2mer": "aa_2mer",
        "aa_3mer": "aa_3mer",
        "nt_1mer": "nt_1mer",
        "nt_2mer": "nt_2mer",
        "nt_3mer": "nt_3mer",
        "nt_4mer": "nt_4mer",
        "nt_5mer": "nt_5mer",
        "cds_1mer": "cds_1mer",
        "cds_2mer": "cds_2mer",
        "cds_3mer": "cds_3mer",
        "cds_4mer": "cds_4mer",
        "cds_5mer": "cds_5mer"
    }

    # Chemical features are kinda custom.
    # TODO: put these in the h5 file.
    if feature_type == "chemical":
        metadata_df = pd.read_hdf(feature_path, key=key_dict["metadata"])
        ngenes = metadata_df.drop_duplicates()["number_of_genes"]
        nt1_features_df = pd.read_hdf(feature_path, key=key_dict["nt_1mer"])
        cds1_features_df = pd.read_hdf(feature_path, key=key_dict["cds_1mer"])
        aa1_features_df = pd.read_hdf(feature_path, key=key_dict["aa_1mer"])
        gc = calc_gc(nt1_features_df)
        aa_features = calc_aa_features(aa1_features_df)
        rna_features = calc_rna_features(cds1_features_df)

        output["features"] = pd.concat(
            [gc, ngenes, aa_features, rna_features], axis=1).dropna(axis=0)
        return output

    err_msg = "please use KO (all KO counts), WGE (whole genome embedding), OGSE (oxygen gene set embedding), aa_1mer, aa_2mer, or aa_3mer, nt_1mer, nt_2mer, nt_3mer, nt_4mer, or nt_5mer, or cds-1-5 mer"
    assert feature_type in key_dict, err_msg
    key = key_dict[feature_type]
    feature_matrix = pd.read_hdf(feature_path,key=key)
    
    # if the feature matrix is from the nt class, then remove the first keep only the string after the first "_" character in the index for the labels dataframe
    if feature_type.startswith("nt_"):
        output["labels"].index = ["_".join(x.split("_")[1:]) for x in output["labels"].index.tolist()]

    output["features"] = feature_matrix
    return output


def load_ko2ec():
    p = path.join(ASSET_PATH, "mappings/keggOrthogroupsToECnumbers.07Feb2023.csv")
    return pd.read_csv(p, index_col=0)


def load_oxygen_kos():
    p = path.join(ASSET_PATH, "mappings/ko_groups.oxygenAssociated.07Feb2023")
    return pd.read_csv(p, index_col=0)


def load_validation_data(feature_type="KO"):
    output = {"features":[],"labels":[]}
    feature_path = path.join(ASSET_PATH, "validation/features/")
    jb_labels_path = path.join(ASSET_PATH, "validation/labels/Jablonska_Labels.07Feb2023.csv")
    labels = pd.read_csv(jb_labels_path, index_col=0)
    output["labels"] = labels

    feature_fname_dict = {
        'KO': "Jablonska_FS.KOCounts.07Feb2023.csv",
        'embedding.genome': "Jablonska_FS.WGE.07Feb2023.csv",
        'embedding.geneset.oxygen': "Jablonska_FS.OGSE.07Feb2023.csv",
        'metadata': "Jablonska_FS.AF.07Feb2023.csv",
        'aa_1mer': "Jablonska_aa_1_mer.16Jul2023.csv",
        'aa_2mer': "Jablonska_aa_2_mer.16Jul2023.csv",
        'aa_3mer': "Jablonska_aa_3_mer.16Jul2023.csv",
        'nt_1mer': "Jablonska.nucletoide_1mers.19Jul2023.csv",
        'nt_2mer': "Jablonska.nucletoide_2mers.19Jul2023.csv",
        'nt_3mer': "Jablonska.nucletoide_3mers.19Jul2023.csv",
        'nt_4mer': "Jablonska.nucletoide_4mers.19Jul2023.csv",
        'nt_5mer': "Jablonska.nucletoide_5mers.19Jul2023.csv",
        'cds_1mer': "Jablonska_cds_1mer_features.csv",
        'cds_2mer': "Jablonska_cds_2mer_features.csv",
        'cds_3mer': "Jablonska_cds_3mer_features.csv",
        'cds_4mer': "Jablonska_cds_4mer_features.csv",
        'cds_5mer': "Jablonska_cds_5mer_features.csv"
    }
    fill_na = set("KO,embedding.genome,embedding.geneset.oxygen".split(","))

    # Chemical features are kinda custom.
    # TODO: save these as a CSV.
    if feature_type == "chemical":
        metadata_fname = path.join(feature_path, feature_fname_dict["metadata"])
        nt1_fname = path.join(feature_path, feature_fname_dict["nt_1mer"])
        cds1_fname = path.join(feature_path, feature_fname_dict["cds_1mer"])
        aa1_fname = path.join(feature_path, feature_fname_dict["aa_1mer"])
        metadata_df = pd.read_csv(metadata_fname, index_col=0).set_index("genome")
        ngenes = metadata_df["number_of_genes"]
        nt1_features_df = pd.read_csv(nt1_fname, index_col=0)
        cds1_features_df = pd.read_csv(cds1_fname, index_col=0)
        aa1_features_df = pd.read_csv(aa1_fname, index_col=0)
        gc = calc_gc(nt1_features_df)  # GC from DNA
        aa_features = calc_aa_features(aa1_features_df)
        rna_features = calc_rna_features(cds1_features_df)

        # remove text after the "." character in all the indices
        rna_features.index = [a for a,_ in rna_features.index.str.split('.')]
        gc.index = [a for a,_ in gc.index.str.split('.')]
        output["features"] = pd.concat(
            [gc, ngenes, aa_features, rna_features], axis=1).dropna(axis=0)
        return output

    feature_matrix = None
    err_string = "please use KO (all KO counts), WGE (whole genome embedding), OGSE (oxygen gene set embedding), aa_1mer, aa_2mer, or aa_3mer, nt_1mer, nt_2mer, nt_3mer, nt_4mer, or nt_5mer"
    assert feature_type in feature_fname_dict, err_string
    fname = feature_fname_dict[feature_type]
    fpath = path.join(feature_path, fname)

    feature_matrix = pd.read_csv(fpath, index_col=0)
    if feature_type in fill_na:
        feature_matrix.fillna(0, inplace=True)
    if feature_type == 'metadata':
        feature_matrix.set_index("genome", inplace=True)
    if feature_type.startswith("nt_"):
        # remove text after the "." character in the index
        feature_matrix.index = [x.split(".")[0] for x in feature_matrix.index.tolist()]
    output["features"] = feature_matrix
    return output
