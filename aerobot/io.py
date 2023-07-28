import pandas as pd
#import h5py
import os
import subprocess as sb
import wget

from os import path

_cwd, _ = path.split(path.abspath(__file__))
ASSET_PATH = path.join(_cwd, 'assets')


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
