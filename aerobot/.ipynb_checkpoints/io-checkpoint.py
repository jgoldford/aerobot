import pandas as pd
#import h5py
import os
import subprocess as sb

asset_path,filename = os.path.split(os.path.abspath(__file__))
asset_path = asset_path + '/assets'

def download_training_data():
    # donwload the training data from google cloud

    fileName = "training_data.tar.gz"
    feature_path = asset_path + "/train/"
    
    if not os.path.exists(feature_path+fileName):
        print("downloading data from google cloud bucket...")
        call = "wget -q -P {fpath} https://storage.googleapis.com/microbe-data/aerobot/{fn}".format(fpath=feature_path,fn=fileName)
        sb.call(call,shell=True)
    if not os.path.exists(feature_path+"training_data.h5"):
        print("extracting feature matrix..")
        sb.call("tar -xvf {fpath}{n} --directory {fpath}".format(fpath=feature_path,n=fileName),shell=True)
    print("done!")
    
def load_training_data(feature_type="KO"):
    output = {"features":[],"labels":[]}
    feature_path = asset_path + "/train/training_data.h5"
    
    #labels = pd.read_csv(asset_path+"/train/labels/Westoby.Trimmed.07Feb2023.csv")
    labels = pd.read_hdf(feature_path,key="labels")
    output["labels"] = labels
    if feature_type == "KO":
        feature_matrix = pd.read_hdf(feature_path,key="KO")
    elif feature_type == "embedding.genome":
        feature_matrix = pd.read_hdf(feature_path,key="WGE")
    elif feature_type == "embedding.geneset.oxygen":
        feature_matrix = pd.read_hdf(feature_path,key="OGSE")
    elif feature_type == "metadata":
        feature_matrix = pd.read_hdf(feature_path,key="AF")        
    else:
        raise Exception("please use KO (all KO counts), WGE (whole genome embedding), or OGSE (oxygen gene set embedding)")
        
    output["features"] = feature_matrix
    return output


def load_ko2ec():
    return pd.read_csv(asset_path + "/mappings/keggOrthogroupsToECnumbers.07Feb2023.csv",index_col=0)

def load_oxygen_kos():
    return pd.read_csv(asset_path + "/mappings/ko_groups.oxygenAssociated.07Feb2023",index_col=0)

def load_validation_data(feature_type="KO"):
    output = {"features":[],"labels":[]}
    feature_path = asset_path + "/validation/features/"
    labels = pd.read_csv(asset_path+"/validation/labels/Jablonska_Labels.07Feb2023.csv",index_col=0)
    output["labels"] = labels
    if feature_type == "KO":
        feature_matrix = pd.read_csv(feature_path + "Jablonska_FS.KOCounts.07Feb2023.csv",index_col=0)
    elif feature_type == "embedding.genome":
        feature_matrix = pd.read_csv(feature_path + "Jablonska_FS.WGE.07Feb2023.csv",index_col=0)
    elif feature_type == "embedding.geneset.oxygen":
        feature_matrix = pd.read_csv(feature_path + "Jablonska_FS.OGSE.07Feb2023.csv",index_col=0)
    elif feature_type == "metadata":
        feature_matrix = pd.read_csv(feature_path + "Jablonska_FS.AF.07Feb2023.csv",index_col=0)
    else:
        raise Exception("please use KO (all KO counts), WGE (whole genome embedding), or OGSE (oxygen gene set embedding)")
    output["features"] = feature_matrix
    return output