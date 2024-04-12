#%%
import pandas as pd
from typing import List
from collections.abc import Callable

def load_HBDxBase():
    version = '_v4'
    HBDxBase_file = f'../../references/HBDxBase/HBDxBase_all{version}.csv'
    HBDxBase_df = pd.read_csv(HBDxBase_file, index_col=0)
    HBDxBase_df.loc[:,'precursor_bins'] = (HBDxBase_df.precursor_length/25).astype(int)
    return HBDxBase_df

def compute_dynamic_bin_size(precursor_len:int, name:str=None, min_bin_size:int=20, max_bin_size:int=30) -> List[int]:
    '''
    This function splits precursor to bins of size max_bin_size
    if the last bin is smaller than min_bin_size, it will split the precursor to bins of size max_bin_size-1
    This process will continue until the last bin is larger than min_bin_size.
    if the min bin size is reached and still the last bin is smaller than min_bin_size, the last two bins will be merged.
    so the maximimum bin size possible would be min_bin_size+(min_bin_size-1) = 39
    '''
    def split_precursor_to_bins(precursor_len,max_bin_size):
        '''
        This function splits precursor to bins of size max_bin_size
        '''
        precursor_bin_lens = []
        for i in range(0, precursor_len, max_bin_size):
            if i+max_bin_size < precursor_len:
                precursor_bin_lens.append(max_bin_size)
            else:
                precursor_bin_lens.append(precursor_len-i)
        return precursor_bin_lens

    if precursor_len < min_bin_size:
        return [precursor_len]
    else:
        precursor_bin_lens = split_precursor_to_bins(precursor_len,max_bin_size)
        reduced_len = max_bin_size-1
        while precursor_bin_lens[-1] < min_bin_size:
            precursor_bin_lens = split_precursor_to_bins(precursor_len,reduced_len)
            reduced_len -= 1
            if reduced_len < min_bin_size:
                #add last two bins together
                precursor_bin_lens[-2] += precursor_bin_lens[-1]
                precursor_bin_lens = precursor_bin_lens[:-1]
                break

        return precursor_bin_lens
    
def get_bin_no_from_pos(precursor_len:int,position:int,name:str=None,min_bin_size:int=20,max_bin_size:int=30) -> int:
    '''
    This function returns the bin number of a position in a precursor
    bins start from 1
    '''
    precursor_bin_lens = compute_dynamic_bin_size(precursor_len=precursor_len,name=name,min_bin_size=min_bin_size,max_bin_size=max_bin_size)
    bin_no = 0
    for i,bin_len in enumerate(precursor_bin_lens):
        if position < bin_len:
            bin_no = i
            break
        else:
            position -= bin_len
    return bin_no+1

def get_bin_with_max_overlap(row) -> int:
    '''
    This function returns the bin number of a fragment that overlaps the most with the fragment
    '''
    precursor_len = row.precursor_length
    start_frag_pos = row.ref_start
    frag_len = row.seq_length
    name = row.precursor_name_full
    min_bin_size = 20
    max_bin_size = 30
    precursor_bin_lens = compute_dynamic_bin_size(precursor_len=precursor_len,name=name,min_bin_size=min_bin_size,max_bin_size=max_bin_size)
    bin_no = 0
    for i,bin_len in enumerate(precursor_bin_lens):
        if start_frag_pos < bin_len:
            #get overlap with curr bin
            overlap = min(bin_len-start_frag_pos,frag_len)

            if overlap > frag_len/2:
                bin_no = i
            else:
                bin_no = i+1
            break

        else:
            start_frag_pos -= bin_len
    #get bin start and bin end
    bin_start,bin_end = sum(precursor_bin_lens[:bin_no]),sum(precursor_bin_lens[:bin_no+1])
    row['bin_start'] = bin_start
    row['bin_end'] = bin_end
    row['subclass_name'] = name + '_bin-' + str(bin_no+1)
    row['precursor_bins'] = len(precursor_bin_lens)
    row['subclass_name_bin_pos'] = name + '_binpos-' + str(bin_start) + ':' + str(bin_end)
    return row
    
def convert_bin_to_pos(precursor_len:int,bin_no:int,bin_function:Callable=compute_dynamic_bin_size,name:str=None,min_bin_size:int=20,max_bin_size:int=30):
    '''
    This function returns the start and end position of a bin
    '''
    precursor_bin_lens = bin_function(precursor_len=precursor_len,name=name,min_bin_size=min_bin_size,max_bin_size=max_bin_size)
    start_pos = 0
    end_pos = 0
    for i,bin_len in enumerate(precursor_bin_lens):
        if i+1 == bin_no:
            end_pos = start_pos+bin_len
            break
        else:
            start_pos += bin_len
    return start_pos,end_pos

#main
if __name__ == '__main__':
    #read hbdxbase
    HBDxBase_df = load_HBDxBase()
    min_bin_size = 20
    max_bin_size = 30
    #select indices of precurosrs that include 'rRNA' but not 'pseudo'
    rRNA_df = HBDxBase_df[HBDxBase_df.index.str.contains('rRNA') * ~HBDxBase_df.index.str.contains('pseudo')]

    #get bin of index 1
    bins = compute_dynamic_bin_size(len(rRNA_df.iloc[0].sequence),rRNA_df.iloc[0].name,min_bin_size,max_bin_size)
    bin_no = get_bin_no_from_pos(len(rRNA_df.iloc[0].sequence),name=rRNA_df.iloc[0].name,position=1)
    annotation_bin = get_bin_with_max_overlap(len(rRNA_df.iloc[0].sequence),start_frag_pos=1,frag_len=50,name=rRNA_df.iloc[0].name)

# %%
