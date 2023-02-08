import pandas as pd
import numpy as np
import requests
import re
import os
from Bio import SeqIO
import subprocess

# define asset path
asset_path,filename = os.path.split(os.path.abspath(__file__))
asset_path = asset_path + '/assets'


def ko2genes(ko):
    """
    produces a list of gene ID's from a kegg KO number using the KEGG REST API
    """
    if ko is None:
        url = 'http://rest.kegg.jp/link/genes/ko'
    else:
        url = 'http://rest.kegg.jp/link/genes/' + ko
    r = requests.get(url)

    try:
        df = pd.DataFrame([x.split('\t') for x in r.text.split('\n')],columns=['ko','gene'])
        df = df.iloc[0:-1]
    except:
        df = None
    return df

def cpd2ec(cpd):
    """
    returns a dataframe of EC's using or producing an input compound using the KEGG REST API
    """
    if cpd is None:
        url = 'http://rest.kegg.jp/link/ec/cpd'
    else:
        url = 'http://rest.kegg.jp/link/ec/cpd:' + cpd
    r = requests.get(url)
    try:
        df = pd.DataFrame([x.split('\t') for x in r.text.split('\n')],columns=['cpd','ec'])
        df = df.iloc[0:-1]
    except:
        df = None
    return df


def download_aa_seqs(geneList,fileName,batchSize):
    """
    downloads amino acid sequences from a gene list using the KEGG REST API
    """
    geneList_chunks = [geneList[i * batchSize:(i + 1) * batchSize] for i in range((len(geneList) + batchSize - 1) // batchSize )]
    with open(fileName,'w') as fastafile:
        for sglist in geneList_chunks:
            url =  'http://rest.kegg.jp/get/' + "+".join(sglist) + '/aaseq'
            fseqs = requests.get(url).text
            fastafile.write(fseqs)

def download_nt_seqs(geneList,fileName,batchSize):
    """
    downloads nucleotide sequences from a gene list using the KEGG REST API
    """
    geneList_chunks = [geneList[i * batchSize:(i + 1) * batchSize] for i in range((len(geneList) + batchSize - 1) // batchSize )]
    with open(fileName,'w') as fastafile:
        for sglist in geneList_chunks:
            url =  'http://rest.kegg.jp/get/' + "+".join(sglist) + '/ntseq'
            fseqs = requests.get(url).text
            fastafile.write(fseqs)


def link(field1,field2):
    """
    returns a link between two fields using the KEGG REST API
    e.g. link("ko","rn") returns all ko to reaction associations
    """
    url = 'http://rest.kegg.jp/link/' + field1 + '/' + field2
    r = requests.get(url)
    try:
        df = pd.DataFrame([x.split('\t') for x in r.text.split('\n')],columns=['f1','f2'])
        df = df.iloc[0:-1]
    except:
        df = None
    return df


def list(field1):
    """
    returns a list of entries using the KEGG REST API
    e.g. list("ko") returns all list of all KOs in KEGG
    """   
    url = 'http://rest.kegg.jp/list/' + field1
    r = requests.get(url)
    try:
        # columns=[field1,'description']
        df = pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
        df = df.iloc[0:-1]
    except:
        df = None
    return df

def convert_genes_to_db(geneList,db,batchSize):
    dfs = []
    geneList_chunks = [geneList[i * batchSize:(i + 1) * batchSize] for i in range((len(geneList) + batchSize - 1) // batchSize )]
    for sglist in geneList_chunks:
        url =  'http://rest.kegg.jp/conv/' + db + '/' + "+".join(sglist) 
        r = requests.get(url)
        try:
            # columns=[field1,'description']
            df = pd.DataFrame([x.split('\t') for x in r.text.split('\n')])
            df = df.iloc[0:-1]
        except:
            df = None
        dfs.append(df)
    dfs = pd.concat(dfs,axis=0)
    return dfs
   