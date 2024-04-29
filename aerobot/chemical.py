'''Code for computing chemical features from nucleotide, CDS, and amino acid 1mers.'''
import pandas as pd
import os
import argparse

CWD, _ = os.path.split(os.path.abspath(__file__))
ASSET_PATH = os.path.join(CWD, 'assets')


CANONICAL_NTS = ['A', 'C', 'G', 'T']
GC_NTS = ['G', 'C']
NT_NOSC_DF = pd.read_csv(ASSET_PATH + '/nt_nosc.csv')
RNA_NOSC_DF = NT_NOSC_DF[NT_NOSC_DF.type == 'RNA']
RNA_NOSC_DF = RNA_NOSC_DF.set_index('letter_code')
RNA_NC = RNA_NOSC_DF.NC
RNA_ZC = RNA_NOSC_DF.NOSC
CANONICAL_NTS_ALL = RNA_NOSC_DF.index.unique().tolist() # Includes DNA and RNA names
AA_NOSC_DF = pd.read_csv(ASSET_PATH + '/aa_nosc.csv', index_col=0)
AA_NC = AA_NOSC_DF.NC
AA_ZC = AA_NOSC_DF.NOSC
CANONICAL_AAS = AA_NOSC_DF.index.tolist()


# TODO: Might be useful to comment this.
def get_rna_features(nt_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate chemical features of the RNA coding sequences. Calculates the formal 
    C oxidation state of mRNAs, as well as the C, N, O and S content. 

    :param nt_1mer_df: A DataFrame containing the counts of single nucleotides, either RNA or DNA. Assumed to single stranded.
    :return: A pd.Series containing the formal C oxidation state of the RNA coding sequences.
    '''
    my_nts = sorted(set(nt_1mer_df.columns).intersection(CANONICAL_NTS_ALL))
    canonical_data = nt_1mer_df[my_nts]
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


def get_gc_content(nt_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate the GC content of the canonical nucleotides.

    :param nt_1mer_df: A DataFrame containing the counts of single nucleotides, either RNA or DNA. Assumed to single stranded.
    :return: A pd.Series containing the GC content of each genome.
    '''
    canonical_data = nt_1mer_df[CANONICAL_NTS]
    gc_content = canonical_data[GC_NTS].sum(axis=1) / canonical_data.sum(axis=1)
    gc_content.name = 'gc_content'
    return gc_content


def get_aa_features(aa_1mer_df:pd.DataFrame) -> pd.Series:
    '''Calculate chemical features of the coding sequences.Calculates the formal C oxidation state 
    of mRNAs, as well as the C, N, O and S content. 

    :param aa_1mer_df: A DataFrame containing the counts of single amino acids.
    :return: A pd.Series containing the formal C oxidation state.
    '''
    aas = sorted(set(aa_1mer_df.columns).intersection(CANONICAL_AAS))
    canonical_data = aa_1mer_df[aas]
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


def chemical_get_features(metadata_df:pd.DataFrame=None, cds_1mer_df:pd.DataFrame=None, aa_1mer_df:pd.DataFrame=None, nt_1mer_df=None) -> pd.DataFrame:
    '''Compute chemical features using other feature DataFrames and the metadata.

    :param metadata_df: DataFrame containing the gene metadata.
    :param nt_1mer_df: DataFrame containing the nt_1mer feature data.
    :param aa_1mer_df: DataFrame containing the aa_1mer feature data.
    :param cds_1mer_df: DataFrame containing the cds_1mer feature data.
    :return: A DataFrame containing the chemical feature data.
    '''
    n_genes = metadata_df.drop_duplicates()['number_of_genes']
    gc_content = get_gc_content(nt_1mer_df)
    aa_features = get_aa_features(aa_1mer_df)
    rna_features = get_rna_features(cds_1mer_df)

    return pd.concat([gc_content, n_genes, aa_features, rna_features], axis=1).dropna(axis=0)
