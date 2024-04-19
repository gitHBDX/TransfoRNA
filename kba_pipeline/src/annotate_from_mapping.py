######################################################################################################
# annotate sequences based on mapping results
######################################################################################################
#%%
import os
import logging

import numpy as np
import pandas as pd
from difflib import get_close_matches
import json

from joblib import Parallel, delayed
import multiprocessing


from utils import (fasta2df, fasta2df_subheader,log_time, reverse_complement)
from precursor_bins import get_bin_with_max_overlap


log = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


######################################################################################################
# paths to reference and mapping files
######################################################################################################

version = '_v4'

HBDxBase_csv = f'../../references/HBDxBase/HBDxBase_all{version}.csv'
miRBase_mature_path = '../../references/HBDxBase/miRBase/mature.fa'
mat_miRNA_pos_path = '../../references/HBDxBase/miRBase/hsa_mature_position.txt'

mapped_file = 'seqsmapped2HBDxBase_combined.txt'
unmapped_file = 'tmp_seqs3mm2HBDxBase_pseudo__unmapped.fa'
TE_file = 'tmp_seqsmapped2genome_intersect_TE.txt'
mapped_genome_file = 'seqsmapped2genome_combined.txt'
toomanyloci_genome_file = 'tmp_seqs0mm2genome__toomanyalign.fa'
unmapped_adapter_file = 'tmp_seqs3mm2adapters__unmapped.fa'
unmapped_genome_file = 'tmp_seqs0mm2genome__unmapped.fa'
unmapped_bacterial_file = 'tmp_seqs0mm2bacterial__unmapped.fa'
unmapped_viral_file = 'tmp_seqs0mm2viral__unmapped.fa'


sRNA_anno_file = 'sRNA_anno_from_mapping.csv'
aggreg_sRNA_anno_file = 'sRNA_anno_aggregated_on_seq.csv'



#%%
######################################################################################################
# specific functions
######################################################################################################

@log_time(log)
def extract_general_info(mapping_file):
    # load mapping file
    mapping_df = pd.read_csv(mapping_file, sep='\t', header=None)
    mapping_df.columns = ['tmp_seq_id','reference','ref_start','sequence','other_alignments','mm_descriptors']

    # add precursor length + number of bins that will be used for names
    HBDxBase_df = pd.read_csv(HBDxBase_csv, index_col=0)
    HBDxBase_df = HBDxBase_df[['precursor_length','precursor_bins','pseudo_class']].reset_index()
    HBDxBase_df.rename(columns={'index': "reference"}, inplace=True)
    mapping_df = mapping_df.merge(HBDxBase_df, left_on='reference', right_on='reference', how='left')

    # extract information
    mapping_df.loc[:,'mms'] = mapping_df.mm_descriptors.fillna('').str.count('>')
    mapping_df.loc[:,'mm_descriptors'] = mapping_df.mm_descriptors.str.replace(',', ';')
    mapping_df.loc[:,'small_RNA_class_annotation'] = mapping_df.reference.str.split('|').str[0]
    mapping_df.loc[:,'subclass_type'] = mapping_df.reference.str.split('|').str[2]
    mapping_df.loc[:,'precursor_name_full'] = mapping_df.reference.str.split('|').str[1].str.split('|').str[0]
    mapping_df.loc[:,'precursor_name'] = mapping_df.precursor_name_full.str.split('__').str[0].str.split('|').str[0]
    mapping_df.loc[:,'seq_length'] = mapping_df.sequence.apply(lambda x: len(x))
    mapping_df.loc[:,'ref_end'] = mapping_df.ref_start +  mapping_df.seq_length - 1
    mapping_df.loc[:,'mitochondrial'] = np.where(mapping_df.reference.str.contains(r'(\|MT-)|(12S)|(16S)'), 'mito', 'nuclear')

    # get non-templated 3' polyA and polyT tails
    ref_end_df = mapping_df.mm_descriptors.str.split(';', expand=True)
    if ref_end_df.shape[1] == 1:
        tail_df = pd.concat([(mapping_df.seq_length - 1)],axis=1)
        tail_df.columns = [0]
    elif ref_end_df.shape[1] == 2:
        tail_df = pd.concat([(mapping_df.seq_length - 2),(mapping_df.seq_length - 1)],axis=1)
        tail_df.columns = [0,1]
    elif ref_end_df.shape[1] == 3:
        tail_df = pd.concat([(mapping_df.seq_length - 3),(mapping_df.seq_length - 2),(mapping_df.seq_length - 1)],axis=1)
        tail_df.columns = [0,1,2]
    ref_end_mask = ref_end_df.apply(lambda x: x.str.split(':').str[0]).fillna(0).astype(int) == tail_df
    ref_end_df = ref_end_df.apply(lambda x: x.str.split('>').str[1])[ref_end_mask]
    ref_end = ref_end_df.fillna('').apply(lambda x: ''.join(x),axis=1)
    ref_end[~ref_end.str.match(r'(^A$)|(^AA$)|(^AAA$)|(^T)$|(^TT$)|(^TTT$)')] = ''
    mapping_df.loc[:,'polyAT'] = ref_end
    
    return mapping_df


#%%
@log_time(log)
def tRNA_annotation(mapping_df):
    """Extract tRNA specific annotation from mapping.
    """
    # keep only tRNA leader/trailer with right cutting sites (+/- 5nt)
    # leader
    tRF_leader_df = mapping_df[mapping_df['subclass_type'] == 'leader_tRF']
    # assign as misc-leader-tRF if exceeding defined cutting site range
    tRF_leader_df.loc[:,'subclass_type'] = np.where((tRF_leader_df.ref_start + tRF_leader_df.sequence.apply(lambda x: len(x))).between(45, 55, inclusive='both'), 'leader_tRF', 'misc-leader-tRF')

    # trailer
    tRF_trailer_df = mapping_df[mapping_df['subclass_type'] == 'trailer_tRF']
    # assign as misc-trailer-tRF if exceeding defined cutting site range
    tRF_trailer_df.loc[:,'subclass_type'] = np.where(tRF_trailer_df.ref_start.between(0, 5, inclusive='both'), 'trailer_tRF', 'misc-trailer-tRF')

    # define tRF subclasses (leader_tRF and trailer_tRF have been assigned previously)
    # NOTE: allow more flexibility at ends (similar to miRNA annotation)
    tRNAs_df = mapping_df[((mapping_df['small_RNA_class_annotation'] == 'tRNA') & mapping_df['subclass_type'].isna())]
    tRNAs_df.loc[((tRNAs_df.ref_start < 3) & (tRNAs_df.seq_length >= 30)),'subclass_type'] = '5p-tR-half'
    tRNAs_df.loc[((tRNAs_df.ref_start < 3) & (tRNAs_df.seq_length < 30)),'subclass_type'] = '5p-tRF'
    tRNAs_df.loc[(((tRNAs_df.precursor_length - (tRNAs_df.ref_end + 1)) < 6) & (tRNAs_df.seq_length >= 30)),'subclass_type'] = '3p-tR-half'
    tRNAs_df.loc[(((tRNAs_df.precursor_length - (tRNAs_df.ref_end + 1)).between(3,6,inclusive='neither')) & (tRNAs_df.seq_length < 30)),'subclass_type'] = '3p-tRF'
    tRNAs_df.loc[(((tRNAs_df.precursor_length - (tRNAs_df.ref_end + 1)) < 3) & (tRNAs_df.seq_length < 30)),'subclass_type'] = '3p-CCA-tRF' 
    tRNAs_df.loc[tRNAs_df.subclass_type.isna(),'subclass_type'] = 'misc-tRF'
    # add ref_iso flag
    tRNAs_df['tRNA_ref_iso'] = np.where(
        (
            (tRNAs_df.ref_start == 0) 
            | ((tRNAs_df.ref_end + 1) == tRNAs_df.precursor_length) 
            | ((tRNAs_df.ref_end + 1) == (tRNAs_df.precursor_length - 3))
        ), 'reftRF', 'isotRF'
    )
    # concat tRNA, leader & trailer dfs
    tRNAs_df = pd.concat([tRNAs_df, tRF_leader_df, tRF_trailer_df],axis=0)
    # adjust precursor name and create tRNA name
    tRNAs_df['precursor_name'] = tRNAs_df.precursor_name.str.extract(r"((tRNA-...-...)|(MT-..)|(tRX-...-...)|(tRNA-i...-...))", expand=True)[0]
    tRNAs_df['subclass_name'] = tRNAs_df.subclass_type + '__' + tRNAs_df.precursor_name
    
    return tRNAs_df

#%%
@log_time(log)
def miRNA_annotation(mapping_df):
    """Extract miRNA specific annotation from mapping. IsomiR rules based on Tomasello et al. 2021 are applied.
    """
    # load positions of mature miRNAs within precursor
    miRNA_pos_df = pd.read_csv(mat_miRNA_pos_path, sep='\t')
    miRNA_pos_df.drop(columns=['precursor_length'], inplace=True)

    miRNAs_df = mapping_df[mapping_df.small_RNA_class_annotation == 'miRNA']
    miRNAs_df = miRNAs_df.merge(miRNA_pos_df, left_on='precursor_name_full', right_on='name_precursor', how='left')
    # drop seqs that are not in range +/- 2nt of mature start
    miRNAs_df = miRNAs_df[miRNAs_df.ref_start.between(miRNAs_df.mature_start-2, miRNAs_df.mature_start+2, inclusive='both')]
    # drop seqs with mismatch unless A>G or C>T in seed region (= position 0-8) or 3' polyA/polyT
    miRNAs_df = miRNAs_df[
        (
            ((miRNAs_df.mms == miRNAs_df.polyAT.apply(lambda x: len(x))) & ~miRNAs_df.polyAT.isna()) 
            | (miRNAs_df.mm_descriptors.str.split(';',expand=True).apply(lambda x: x.str.contains(r'(^[0-8]:A>G)|(^[0-8]:C>T)',na=True)).all(axis=1)) 
        )]
    # add difference to mature position (5' end, 3' end)
    miRNAs_df['miRNA_mature_diff'] = '5p:' + (miRNAs_df.ref_start - miRNAs_df.mature_start).astype(int).astype(str) + ',3p:' + (miRNAs_df.ref_end - miRNAs_df.mature_end).astype(int).astype(str) + ',nt3p:' + miRNAs_df.polyAT
    # add ref_iso flag
    miRNAs_df['miRNA_ref_iso'] = np.where(
        (
            (miRNAs_df.ref_start == miRNAs_df.mature_start) 
            & (miRNAs_df.ref_end == miRNAs_df.mature_end) 
            & (miRNAs_df.mms == 0)
        ), 'refmiR', 'isomiR'
    )
    # add subclass (NOTE: in cases where subclass is not part of mature name, use position relative to precursor half to define group )
    miRNAs_df['miRNA__subclass'] = np.where(miRNAs_df.name_mature.str.endswith('5p'), '5p', np.where(miRNAs_df.name_mature.str.endswith('3p'), '3p', 'tbd'))
    miRNAs_df.loc[((miRNAs_df.miRNA__subclass == 'tbd') & (miRNAs_df.mature_start < miRNAs_df.precursor_length/2)), 'miRNA__subclass'] = '5p'
    miRNAs_df.loc[((miRNAs_df.miRNA__subclass == 'tbd') & (miRNAs_df.mature_start >= miRNAs_df.precursor_length/2)), 'miRNA__subclass'] = '3p'
    
    # load mature miRNA sequences from miRBase
    miRBase_mature_df = fasta2df_subheader(miRBase_mature_path,0)
    # subset to human miRNAs
    miRBase_mature_df = miRBase_mature_df.loc[miRBase_mature_df.index.str.contains('hsa-'),:]
    miRBase_mature_df.index = miRBase_mature_df.index.str.replace('hsa-','')
    miRBase_mature_df.reset_index(inplace=True)
    miRBase_mature_df.columns = ['name_mature','ref_miR_seq']
    # add 'ref_miR_seq' 
    miRNAs_df = miRNAs_df.merge(miRBase_mature_df, left_on='name_mature', right_on='name_mature', how='left')
    
    miRNAs_df['subclass_name'] = miRNAs_df.name_mature
    miRNAs_df = miRNAs_df[list(mapping_df.columns) + ['miRNA__subclass','subclass_name','miRNA_ref_iso','miRNA_mature_diff','ref_miR_seq']]

    # set all other miRNA hairpin maps as misc-miRNA
    miRNAs_df = mapping_df[mapping_df.small_RNA_class_annotation == 'miRNA'].merge(miRNAs_df, left_on=list(mapping_df.columns), right_on=list(mapping_df.columns), how='left')
    miRNAs_df['miRNA__subclass'] = miRNAs_df.miRNA__subclass.fillna('misc')
    miRNAs_df.loc[miRNAs_df.subclass_name.isna(),'subclass_name'] = miRNAs_df.precursor_name_full
    miRNAs_df['subclass_type'] = miRNAs_df.miRNA__subclass
    miRNAs_df = miRNAs_df.drop(columns=['miRNA__subclass'])

    return miRNAs_df


#%%
######################################################################################################
# annotation of other sRNA classes
######################################################################################################
def get_bin_with_max_overlap_parallel(df):
    return df.apply(get_bin_with_max_overlap, axis=1)

def applyParallel(df, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for group in np.array_split(df,30))
    return pd.concat(retLst)


@log_time(log)
def other_sRNA_annotation_new_binning(mapping_df):
    """Generate subclass_name for non-tRNA/miRNA sRNAs by precursor-binning.
    New binning approach: bin size is dynamically determined by the precursor length. Assignments are based on the bin with the highest overlap.
    """

    other_sRNAs_df = mapping_df[~((mapping_df.small_RNA_class_annotation == 'miRNA') | (mapping_df.small_RNA_class_annotation == 'tRNA'))]
    
    #create empty columns; bin start and bin end
    other_sRNAs_df['bin_start'] = ''
    other_sRNAs_df['bin_end'] = ''
    
    other_sRNAs_df = applyParallel(other_sRNAs_df, get_bin_with_max_overlap_parallel)
    
    return other_sRNAs_df


#%%
@log_time(log)
def extract_sRNA_class_specific_info(mapping_df):
    tRNAs_df = tRNA_annotation(mapping_df)
    miRNAs_df = miRNA_annotation(mapping_df)
    other_sRNAs_df = other_sRNA_annotation_new_binning(mapping_df)
    
    # add miRNA columns
    tRNAs_df[['miRNA_ref_iso', 'miRNA_mature_diff', 'ref_miR_seq']] = pd.DataFrame(columns=['miRNA_ref_iso', 'miRNA_mature_diff', 'ref_miR_seq'])
    other_sRNAs_df[['miRNA_ref_iso', 'miRNA_mature_diff', 'ref_miR_seq']] = pd.DataFrame(columns=['miRNA_ref_iso', 'miRNA_mature_diff', 'ref_miR_seq'])
    
    # re-concat sRNA class dfs
    sRNA_anno_df = pd.concat([miRNAs_df, tRNAs_df, other_sRNAs_df],axis=0)

    # TEST if alignments were lost or duplicated
    assert (len(mapping_df) == len(sRNA_anno_df)), "alignments were lost or duplicated" 
    
    return sRNA_anno_df

#%%
def get_nth_nt(row):
    return row['sequence'][int(row['PTM_position_in_seq'])-1]



#%%
@log_time(log)
def aggregate_info_per_seq(sRNA_anno_df):
    # fillna of 'subclass_name_bin_pos' with 'subclass_name'
    sRNA_anno_df['subclass_name_bin_pos'] = sRNA_anno_df['subclass_name_bin_pos'].fillna(sRNA_anno_df['subclass_name'])
    # get aggregated info per seq 
    aggreg_per_seq_df = sRNA_anno_df.groupby(['sequence']).agg({'small_RNA_class_annotation': lambda x: ';'.join(sorted(x.unique())), 'pseudo_class': lambda x: ';'.join(x.astype(str).sort_values(ascending=True).unique()), 'subclass_type': lambda x: ';'.join(x.astype(str).sort_values(ascending=True).unique()), 'subclass_name': lambda x: ';'.join(sorted(x.unique())), 'subclass_name_bin_pos': lambda x: ';'.join(sorted(x.unique())), 'precursor_name_full': lambda x: ';'.join(sorted(x.unique())), 'mms': lambda x: ';'.join(x.astype(str).sort_values(ascending=True).unique()), 'reference': lambda x: len(x), 'mitochondrial': lambda x: ';'.join(x.astype(str).sort_values(ascending=True).unique()), 'ref_miR_seq': lambda x: ';'.join(x.fillna('').unique())})
    aggreg_per_seq_df.ref_miR_seq = aggreg_per_seq_df.ref_miR_seq.str.replace(r';$','')
    aggreg_per_seq_df['mms'] = aggreg_per_seq_df['mms'].astype(int)

    # re-add 'miRNA_ref_iso'
    refmir_df = sRNA_anno_df[['sequence','miRNA_ref_iso','tRNA_ref_iso']]
    refmir_df.drop_duplicates('sequence', inplace=True)
    refmir_df.set_index('sequence', inplace=True)
    aggreg_per_seq_df = aggreg_per_seq_df.merge(refmir_df, left_index=True, right_index=True, how='left')

    # TEST if sequences were lost
    assert (len(aggreg_per_seq_df) == len(sRNA_anno_df.sequence.unique())), "sequences were lost by aggregation" 

    # load unmapped seqs, if it exits
    if os.path.exists(unmapped_file):
        unmapped_df = fasta2df(unmapped_file)
        unmapped_df = pd.DataFrame(data='no_annotation', index=unmapped_df.sequence, columns=aggreg_per_seq_df.columns)
        unmapped_df['mms'] = np.nan
        unmapped_df['reference'] = np.nan
        unmapped_df['pseudo_class'] = True # set no annotation as pseudo_class

        # merge mapped and unmapped
        annotation_df = pd.concat([aggreg_per_seq_df,unmapped_df])
    else:
        annotation_df = aggreg_per_seq_df.copy()

    # load mapping to genome file
    mapping_genome_df = pd.read_csv(mapped_genome_file, index_col=0, sep='\t', header=None)
    mapping_genome_df.columns = ['strand','reference','ref_start','sequence','other_alignments','mm_descriptors']
    mapping_genome_df = mapping_genome_df[['strand','reference','ref_start','sequence','other_alignments']]

    # use reverse complement of 'sequence' for 'strand' == '-'
    mapping_genome_df.loc[:,'sequence'] = np.where(mapping_genome_df.strand == '-', mapping_genome_df.sequence.apply(lambda x: reverse_complement(x)), mapping_genome_df.sequence)

    # get aggregated info per seq
    aggreg_per_seq__genome_df = mapping_genome_df.groupby('sequence').agg({'reference': lambda x: ';'.join(sorted(x.unique())), 'other_alignments': lambda x: len(x)})
    aggreg_per_seq__genome_df['other_alignments'] = aggreg_per_seq__genome_df['other_alignments'].astype(int)

    # number of genomic loci
    genomic_loci_df = pd.DataFrame(mapping_genome_df.sequence.value_counts())
    genomic_loci_df.columns = ['num_genomic_loci_maps']

    # load too many aligments seqs
    if os.path.exists(toomanyloci_genome_file):
        toomanyloci_genome_df = fasta2df(toomanyloci_genome_file)
        toomanyloci_genome_df = pd.DataFrame(data=101, index=toomanyloci_genome_df.sequence, columns=genomic_loci_df.columns)
    else:
        toomanyloci_genome_df = pd.DataFrame(columns=genomic_loci_df.columns)

    # load unmapped seqs
    if os.path.exists(unmapped_genome_file):
        unmapped_genome_df = fasta2df(unmapped_genome_file)
        unmapped_genome_df = pd.DataFrame(data=0, index=unmapped_genome_df.sequence, columns=genomic_loci_df.columns)
    else:
        unmapped_genome_df = pd.DataFrame(columns=genomic_loci_df.columns)

    # concat toomanyloci, unmapped, and genomic_loci
    num_genomic_loci_maps_df = pd.concat([genomic_loci_df,toomanyloci_genome_df,unmapped_genome_df])

    # merge to annotation_df
    annotation_df = annotation_df.merge(num_genomic_loci_maps_df, left_index=True, right_index=True, how='left')
    annotation_df.reset_index(inplace=True)

    # add 'miRNA_seed'
    annotation_df.loc[:,"miRNA_seed"] = np.where(annotation_df.small_RNA_class_annotation.str.contains('miRNA', na=False), annotation_df.sequence.str[1:9], "")

    # TEST if nan values in 'num_genomic_loci_maps'
    assert (annotation_df.num_genomic_loci_maps.isna().any() == False), "nan values in 'num_genomic_loci_maps'" 

    return annotation_df




#%%
@log_time(log)
def get_five_prime_adapter_info(annotation_df, five_prime_adapter):
    adapter_df = pd.DataFrame(index=annotation_df.sequence)

    min_length = 6

    is_prefixed = None
    print("5' adapter affixes:")
    for l in range(0, len(five_prime_adapter) - min_length):
        is_prefixed_l = adapter_df.index.str.startswith(five_prime_adapter[l:])
        print(f"{five_prime_adapter[l:].ljust(30, ' ')}{is_prefixed_l.sum()}")
        adapter_df.loc[adapter_df.index.str.startswith(five_prime_adapter[l:]), "five_prime_adapter_length"] = len(five_prime_adapter[l:])
        if is_prefixed is None:
            is_prefixed = is_prefixed_l
        else:
            is_prefixed |= is_prefixed_l

    print(f"There are {is_prefixed.sum()} prefixed features.")
    print("\n")

    adapter_df['five_prime_adapter_length'] = adapter_df['five_prime_adapter_length'].fillna(0)
    adapter_df['five_prime_adapter_length'] =  adapter_df['five_prime_adapter_length'].astype('int')
    adapter_df['five_prime_adapter_filter'] = np.where(adapter_df['five_prime_adapter_length'] == 0, True, False)
    adapter_df = adapter_df.reset_index()
    
    return adapter_df

#%%
@log_time(log)
def reduce_ambiguity(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce ambiguity by 

    a) using subclass_name of precursor with shortest genomic context, if all other assigned precursors overlap with its genomic region
    
    b) using subclass_name whose bin is at the 5' or 3' end of the precursor

    Parameters
    ----------
    annotation_df : pd.DataFrame
        A DataFrame containing the annotation of the sequences (var)

    Returns
    -------
    pd.DataFrame
        An improved version of the input DataFrame with reduced ambiguity
    """

    # extract ambigious assignments for subclass name
    ambigious_matches_df = annotation_df[annotation_df.subclass_name.str.contains(';',na=False)]
    if len(ambigious_matches_df) == 0:
        print('No ambigious assignments for subclass name found.')
        return annotation_df
    clear_matches_df = annotation_df[~annotation_df.subclass_name.str.contains(';',na=False)]

    # extract required information from HBDxBase
    HBDxBase_all_df = pd.read_csv(HBDxBase_csv, index_col=0)
    bin_dict = HBDxBase_all_df[['precursor_name','precursor_bins']].set_index('precursor_name').to_dict()['precursor_bins']
    sRNA_class_dict = HBDxBase_all_df[['precursor_name','small_RNA_class_annotation']].set_index('precursor_name').to_dict()['small_RNA_class_annotation']
    pseudo_class_dict = HBDxBase_all_df[['precursor_name','pseudo_class']].set_index('precursor_name').to_dict()['pseudo_class']
    sc_type_dict = HBDxBase_all_df[['precursor_name','subclass_type']].set_index('precursor_name').to_dict()['subclass_type']
    genomic_context_bed = HBDxBase_all_df[['chr','start','end','precursor_name','score','strand']]
    genomic_context_bed.columns = ['seq_id','start','end','name','score','strand']
    genomic_context_bed.reset_index(drop=True, inplace=True)
    genomic_context_bed['genomic_length'] = genomic_context_bed.end - genomic_context_bed.start


    def get_overlaps(genomic_context_bed: pd.DataFrame, name: str = None, complement: bool = False) -> list:
        """Get genomic overlap of a given precursor name

        Parameters
        ----------
        genomic_context_bed : pd.DataFrame
            A DataFrame containing genomic locations of precursors in bed format
        with column names: 'chr','start','end','precursor_name','score','strand'
        name : str
            The name of the precursor to get genomic context for
        complement : bool
            If True, return all precursors that do not overlap with the given precursor

        Returns
        -------
        list
            A list containing the precursors in the genomic (anti-)context of the given precursor 
            (including the precursor itself)
        """
        series_OI = genomic_context_bed[genomic_context_bed['name'] == name]
        start = series_OI['start'].values[0]
        end = series_OI['end'].values[0]
        seq_id = series_OI['seq_id'].values[0]
        strand = series_OI['strand'].values[0]

        overlap_df = genomic_context_bed.copy()

        condition = (((overlap_df.start > start) &
                        (overlap_df.start < end)) |
                        ((overlap_df.end > start) &
                        (overlap_df.end < end)) |
                        ((overlap_df.start < start) &
                        (overlap_df.end > start)) |
                        ((overlap_df.start == start) &
                        (overlap_df.end == end)) |
                        ((overlap_df.start == start) &
                        (overlap_df.end > end)) |
                        ((overlap_df.start < start) &
                        (overlap_df.end == end)))
        if not complement:
            overlap_df = overlap_df[condition]
        else:
            overlap_df = overlap_df[~condition]
        overlap_df = overlap_df[overlap_df.seq_id == seq_id]
        if strand is not None:
            overlap_df = overlap_df[overlap_df.strand == strand]
        overlap_list = overlap_df['name'].tolist()
        return overlap_list


    def check_genomic_ctx_of_smallest_prec(precursor_name: str) -> str:
        """Check for a given ambigious precursor assignment (several names separated by ';')
        if all assigned precursors overlap with the genomic region
        of the precursor with the shortest genomic context

        Parameters
        ----------
        precursor_name: str
            A string containing several precursor names separated by ';'

        Returns
        -------
        str
            The precursor suggested to be used instead of the multi assignment, 
            or None if the ambiguity could not be resolved
        """
        assigned_names = precursor_name.split(';')

        tmp_genomic_context = genomic_context_bed[genomic_context_bed.name.isin(assigned_names)]
        # get name of smallest genomic region
        if len(tmp_genomic_context) > 0:
            smallest_name = tmp_genomic_context.name[tmp_genomic_context.genomic_length.idxmin()]
            # check if all assigned names are in overlap of smallest genomic region
            if set(assigned_names).issubset(set(get_overlaps(genomic_context_bed,smallest_name))):
                return smallest_name
            else:
                return None
        else:
            return None
        
    def get_subclass_name(subclass_name: str, short_prec_match_new_name: str) -> str:
        """Get subclass name matching to a precursor name from a ambigious assignment (several names separated by ';')

        Parameters
        ----------
        subclass_name: str
            A string containing several subclass names separated by ';'
        short_prec_match_new_name: str
            The name of the precursor to be used instead of the multi assignment

        Returns
        -------
        str
            The subclass name suggested to be used instead of the multi assignment, 
            or None if the ambiguity could not be resolved
        """
        if short_prec_match_new_name is not None:
            matches = get_close_matches(short_prec_match_new_name,subclass_name.split(';'),cutoff=0.2)
            if matches:
                return matches[0]
            else:
                print(f"Could not find match for {short_prec_match_new_name} in {subclass_name}")
                return subclass_name
        else:
            return None


    def check_end_bins(subclass_name: str) -> str:
        """Check for a given ambigious subclass name assignment (several names separated by ';')
        if ambiguity can be resolved by selecting the subclass name whose bin matches the 3'/5' end of the precursor

        Parameters
        ----------
        subclass_name: str
            A string containing several subclass names separated by ';'

        Returns
        -------
        str
            The subclass name suggested to be used instead of the multi assignment, 
            or None if the ambiguity could not be resolved
        """
        for name in subclass_name.split(';'):
            if '_bin-' in name:
                name_parts = name.split('_bin-')
                if name_parts[0] in bin_dict and bin_dict[name_parts[0]] == int(name_parts[1]):
                    return name
                elif int(name_parts[1]) == 1:
                    return name
        return None


    def adjust_4_resolved_cases(row: pd.Series) -> tuple:
        """For a resolved ambiguous subclass names return adjusted values of 
        precursor_name_full, small_RNA_class_annotation, pseudo_class, and subclass_type 

        Parameters
        ----------
        row: pd.Series
            A row of the var annotation containing the columns 'subclass_name', 'precursor_name_full',
            'small_RNA_class_annotation', 'pseudo_class', 'subclass_type', and 'ambiguity_resolved'

        Returns
        -------
        tuple
            A tuple containing the adjusted values of 'precursor_name_full', 'small_RNA_class_annotation', 
            'pseudo_class', and 'subclass_type' for resolved ambiguous cases and the original values for unresolved cases
        """
        if row.ambiguity_resolved:
            matches_prec = get_close_matches(row.subclass_name, row.precursor_name_full.split(';'), cutoff=0.2)
            if matches_prec:
                return matches_prec[0], sRNA_class_dict[matches_prec[0]], pseudo_class_dict[matches_prec[0]], sc_type_dict[matches_prec[0]]
        return row.precursor_name_full, row.small_RNA_class_annotation, row.pseudo_class, row.subclass_type
    
    
    # resolve ambiguity by checking genomic context of smallest precursor
    ambigious_matches_df['short_prec_match_new_name'] = ambigious_matches_df.precursor_name_full.apply(check_genomic_ctx_of_smallest_prec)
    ambigious_matches_df['short_prec_match_new_name'] = ambigious_matches_df.apply(lambda x: get_subclass_name(x.subclass_name, x.short_prec_match_new_name), axis=1)
    ambigious_matches_df['short_prec_match'] = ambigious_matches_df['short_prec_match_new_name'].notnull()

    # resolve ambiguity by checking if bin matches 3'/5' end of precursor
    ambigious_matches_df['end_bin_match_new_name'] = ambigious_matches_df.subclass_name.apply(check_end_bins)
    ambigious_matches_df['end_bin_match'] = ambigious_matches_df['end_bin_match_new_name'].notnull()

    # check if short_prec_match and end_bin_match are equal in any case
    test_df = ambigious_matches_df[((ambigious_matches_df.short_prec_match == True) & (ambigious_matches_df.end_bin_match == True))]
    if not (test_df.short_prec_match_new_name == test_df.end_bin_match_new_name).all():
        print('Number of cases where short_prec_match is not matching end_bin_match_new_name:',len(test_df[(test_df.short_prec_match_new_name != test_df.end_bin_match_new_name)]))

    # replace subclass_name with short_prec_match_new_name or end_bin_match_new_name
    # NOTE: if short_prec_match and end_bin_match are True, short_prec_match_new_name is used
    ambigious_matches_df['subclass_name'] = ambigious_matches_df.apply(lambda x: x.end_bin_match_new_name if x.end_bin_match == True else x.subclass_name, axis=1)
    ambigious_matches_df['subclass_name'] = ambigious_matches_df.apply(lambda x: x.short_prec_match_new_name if x.short_prec_match == True else x.subclass_name, axis=1)

    # generate column 'ambiguity_resolved' which is True if short_prec_match and/or end_bin_match is True
    ambigious_matches_df['ambiguity_resolved'] = ambigious_matches_df.short_prec_match | ambigious_matches_df.end_bin_match
    print("Ambiguity resolved?\n",ambigious_matches_df.ambiguity_resolved.value_counts(normalize=True))

    # for resolved ambiguous matches, adjust precursor_name_full, small_RNA_class_annotation, pseudo_class, subclass_type
    ambigious_matches_df[['precursor_name_full','small_RNA_class_annotation','pseudo_class','subclass_type']] = ambigious_matches_df.apply(adjust_4_resolved_cases, axis=1, result_type='expand')

    # drop temporary columns
    ambigious_matches_df.drop(columns=['short_prec_match_new_name','short_prec_match','end_bin_match_new_name','end_bin_match'], inplace=True)
    
    # concat with clear_matches_df
    clear_matches_df['ambiguity_resolved'] = False
    improved_annotation_df = pd.concat([clear_matches_df, ambigious_matches_df], axis=0)
    improved_annotation_df = improved_annotation_df.reindex(annotation_df.index)

    return improved_annotation_df

#%%
######################################################################################################
# HICO (=high confidence) annotation
######################################################################################################
@log_time(log)
def add_hico_annotation(annotation_df, five_prime_adapter):
    """For miRNAs only use hico annotation if part of miRBase hico set AND refmiR
    """

    # add 'TE_annotation'
    TE_df = pd.read_csv(TE_file, sep='\t', header=None, names=['sequence','TE_annotation'])
    annotation_df = annotation_df.merge(TE_df, left_on='sequence', right_on='sequence', how='left')

    # add 'bacterial' mapping filter
    bacterial_unmapped_df = fasta2df(unmapped_bacterial_file)
    annotation_df.loc[:,'bacterial'] = np.where(annotation_df.sequence.isin(bacterial_unmapped_df.sequence), False, True)

    # add 'viral' mapping filter
    viral_unmapped_df = fasta2df(unmapped_viral_file)
    annotation_df.loc[:,'viral'] = np.where(annotation_df.sequence.isin(viral_unmapped_df.sequence), False, True)

    # add 'adapter_mapping_filter' column 
    adapter_unmapped_df = fasta2df(unmapped_adapter_file)
    annotation_df.loc[:,'adapter_mapping_filter'] = np.where(annotation_df.sequence.isin(adapter_unmapped_df.sequence), True, False)

    # add filter column 'five_prime_adapter_filter' and column 'five_prime_adapter_length' indicating the length of the prefixed 5' adapter sequence
    adapter_df = get_five_prime_adapter_info(annotation_df, five_prime_adapter)
    annotation_df = annotation_df.merge(adapter_df, left_on='sequence', right_on='sequence', how='left')

    # apply ambiguity reduction
    annotation_df = reduce_ambiguity(annotation_df)

    # add 'single_class_annotation'
    annotation_df.loc[:,'single_class_annotation'] = np.where(annotation_df.small_RNA_class_annotation.str.contains(';',na=True), False, True)

    # add 'single_name_annotation'
    annotation_df.loc[:,'single_name_annotation'] = np.where(annotation_df.subclass_name.str.contains(';',na=True), False, True)

    # add 'hypermapper' for sequences where more than 50 potential mapping references are recorded
    annotation_df.loc[annotation_df.reference > 50,'subclass_name'] = 'hypermapper_' + annotation_df.reference.fillna(0).astype(int).astype(str)
    annotation_df.loc[annotation_df.reference > 50,'subclass_name_bin_pos'] = 'hypermapper_' + annotation_df.reference.fillna(0).astype(int).astype(str)
    annotation_df.loc[annotation_df.reference > 50,'precursor_name_full'] = 'hypermapper_' + annotation_df.reference.fillna(0).astype(int).astype(str)

    annotation_df.loc[:,'mitochondrial'] = np.where(annotation_df.mitochondrial.str.contains('mito',na=False), True, False)

    # add 'hico' 
    annotation_df.loc[:,'hico'] = np.where((
        (annotation_df.mms == 0) 
        & (annotation_df.single_name_annotation == True)
        & (annotation_df.TE_annotation.isna() == True)
        & (annotation_df.bacterial == False)
        & (annotation_df.viral == False)
        & (annotation_df.adapter_mapping_filter == True)
        & (annotation_df.five_prime_adapter_filter == True)
    ), True, False)
    ## NOTE: for miRNAs only use hico annotation if part of refmiR set
    annotation_df.loc[annotation_df.small_RNA_class_annotation == 'miRNA','hico'] = annotation_df.loc[annotation_df.small_RNA_class_annotation == 'miRNA','hico'] & (annotation_df.miRNA_ref_iso == 'refmiR')

    print(annotation_df[annotation_df.single_class_annotation == True].groupby('small_RNA_class_annotation').hico.value_counts())

    return annotation_df




#%%
######################################################################################################
# annotation pipeline
######################################################################################################
@log_time(log)
def main(five_prime_adapter):
    """Executes 'annotate_from_mapping'.

    Uses:

    - HBDxBase_csv
    - miRBase_mature_path
    - mat_miRNA_pos_path

    - mapping_file
    - unmapped_file
    - mapped_genome_file 
    - toomanyloci_genome_file 
    - unmapped_genome_file

    - TE_file
    - unmapped_adapter_file
    - unmapped_bacterial_file
    - unmapped_viral_file
    - five_prime_adapter

    """


    print('-------- extract general information for sequences that mapped to the HBDxBase --------')
    mapped_info_df = extract_general_info(mapped_file)
    print("\n")

    print('-------- extract sRNA class specific information for sequences that mapped to the HBDxBase --------')
    mapped_sRNA_anno_df = extract_sRNA_class_specific_info(mapped_info_df)

    print('-------- save to file --------')
    mapped_sRNA_anno_df.to_csv(sRNA_anno_file)
    print("\n")
    
    print('-------- aggregate information for mapped and unmapped sequences (HBDxBase & human genome) --------')
    sRNA_anno_per_seq_df = aggregate_info_per_seq(mapped_sRNA_anno_df)
    print("\n")

    print('-------- add hico annotation (based on aggregated infos + mapping to viral/bacterial genomes + intersection with TEs) --------')
    sRNA_anno_per_seq_df = add_hico_annotation(sRNA_anno_per_seq_df, five_prime_adapter)
    print("\n")

    print('-------- save to file --------')
    # set sequence as index again
    sRNA_anno_per_seq_df.set_index('sequence', inplace=True)
    sRNA_anno_per_seq_df.to_csv(aggreg_sRNA_anno_file)
    print("\n")

    print('-------- generate subclass_to_annotation dict --------')
    result_df = sRNA_anno_per_seq_df[['subclass_name', 'small_RNA_class_annotation']].copy()
    result_df.reset_index(drop=True, inplace=True)
    result_df.drop_duplicates(inplace=True)
    result_df = result_df[~result_df["subclass_name"].str.contains(";")] 
    subclass_to_annotation = dict(zip(result_df["subclass_name"],result_df["small_RNA_class_annotation"]))
    with open('subclass_to_annotation.json', 'w') as fp:
        json.dump(subclass_to_annotation, fp)

    print('-------- delete tmp files --------')
    os.system("rm *tmp_*")


#%%
