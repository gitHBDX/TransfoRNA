import random
from contextlib import redirect_stdout
from pathlib import Path
from random import randint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.energy import fold_sequences
from ..utils.file import load
from ..utils.tcga_post_analysis_utils import Results_Handler
from ..utils.utils import (get_model,
                           infer_from_pd, prepare_inference_results_tcga,
                           update_config_with_inference_params)
from ..novelty_prediction.id_vs_ood_nld_clf import get_closest_ngbr_per_split
from .seq_tokenizer import SeqTokenizer


class IDModelAugmenter:
    '''
    This class is used to augment the dataset with the predictions of the ID models
    It will first predict the subclasses of the NA set using the ID models
    Then it will compute the levenstein distance between the sequences of the NA set and the closest neighbor in the training set
    If the levenstein distance is less than a threshold, the sequence is considered familiar
    '''
    def __init__(self,df:pd.DataFrame,config:Dict):
        self.df = df
        self.config = config
        self.mapping_dict = load(config['train_config'].mapping_dict_path)


    def predict_transforna_na(self) -> Tuple:
        infer_pd = pd.DataFrame(columns=['Sequence','Net-Label','Is Familiar?'])

        if True:
            inference_config = update_config_with_inference_params(self.config)
            
            #path should be infer_cfg["model_path"] - 2 level + embedds
            path = '/'.join(inference_config['inference_settings']["model_path"].split('/')[:-2])+'/embedds'
            #read threshold
            results:Results_Handler = Results_Handler(path=path,splits=['train','no_annotation'])
            results.get_knn_model()
            threshold = load(results.analysis_path+"/novelty_model_coef")["Threshold"]
            sequences = results.splits_df_dict['no_annotation_df'][results.seq_col].values[:,0]
            with redirect_stdout(None):
                root_dir = Path(__file__).parents[3].absolute()
                inference_config, net = get_model(inference_config, root_dir)
                infer_pd = pd.Series(sequences, name="Sequences").to_frame()
                print(f'predicting sub classes for the NA set by the ID models')
                predicted_labels, logits,gene_embedds_df, attn_scores_pd,all_data, max_len, net = infer_from_pd(inference_config, net, infer_pd, SeqTokenizer)


            prepare_inference_results_tcga(inference_config, predicted_labels, logits, all_data, max_len)
            infer_pd = all_data["infere_rna_seq"]
            
            #compute lev distance for embedds and 
            print('computing levenstein distance for the NA set by the ID models')
            _,_,_,_,_,lev_dist = get_closest_ngbr_per_split(results,'no_annotation')
            
            print(f'num of hico based on entropy novelty prediction is {sum(infer_pd["Is Familiar?"])}')
            infer_pd['Is Familiar?'] = [True if lv<threshold else False for lv in lev_dist]
            infer_pd['Threshold'] = threshold
            print(f'num of new hico based on levenstein distance is {np.sum(infer_pd["Is Familiar?"])}')
            return infer_pd.rename_axis("Sequence").reset_index()
        else:
            print('Could not load predictions from TransfoRNA, check if ID models exist in the desired structure')
            return infer_pd

    def include_id_model_predictions(self):
        pred_df = self.predict_transforna_na()
        set1 = set(self.df.Labels.cat.categories)
        set2 = set(pred_df['Net-Label'].unique())
        self.df['Labels'] = self.df['Labels'].cat.add_categories(set2.difference(set1))

        #get 'is familiar?' sequences
        familiar_seqs = pred_df[pred_df['Is Familiar?'] == True].Sequence.values
        #add new labels to df by selecting only the sequences that are not hico from df
        self.df.loc[familiar_seqs,'Labels'] = pred_df[pred_df['Is Familiar?'] == True]['Net-Label'].values

    def get_augmented_df(self):
        self.include_id_model_predictions()
        return self.df
    
class RecombinedSeqAugmenter:
    '''
    This class is used to augment the dataset with recombined sequences
    recombinations are done by fusing two sequences from the same subclass
    '''
    def __init__(self,df:pd.DataFrame,config:Dict):
        self.df = df
        self.config = config
        
    def create_recombined_seqs(self,fusion_label:str='recombined'):
        # one set of sequences should be generated 
        #1 - n real RNAs recombined together. n is the number of int(subclasses/2)

        #for set 1, sample one sequence from each subclass
        #sample one sequennce from each unique entry in df.Labels
        #select rows where Labels is not None

        #filter rows where Labels is None
        #get unique labels
        unique_labels = self.df.Labels.value_counts()[self.df.Labels.value_counts() >= 1].index.tolist()
        #get one sequence per label in one line
        samples = [self.df[self.df['Labels'] == label].sample(1).index[0] for label in unique_labels]
        #makes number of samples even
        if len(samples) % 2 != 0:
            samples = samples[:-1]
        np.random.shuffle(samples)
        #split samples into two sets
        samples_set1 = samples[:len(samples)//2]
        samples_set2 = samples[len(samples)//2:]
        #create fusion set
        recombined_set = []
        for i in range(len(samples_set1)):
            recombined_seq = samples_set1[i]+samples_set2[i]
            #get index of the first ntd of the second sequence
            recombined_index = len(samples_set1[i])
            #sample a random offset -5 and 5
            offset = randint(-5,5)
            recombined_index += offset
            #sample an int between 18 and 30
            random_half_len = int(randint(18,30)/2) #9 to 15
            #get the sequence from the recombined sequence
            random_seq = recombined_seq[max(0,recombined_index - random_half_len):recombined_index + random_half_len]
            recombined_set.append(random_seq)

        recombined_df = pd.DataFrame(index=recombined_set, data=[f'{fusion_label}']*len(recombined_set)\
            , columns =['Labels'])

        return recombined_df
    
    def get_augmented_df(self):
        recombined_df = self.create_recombined_seqs()
        return recombined_df
    
class RandomSeqAugmenter:
    '''
    This class is used to augment the dataset with random sequences within the same length range as the tcga sequences
    '''
    def __init__(self,df:pd.DataFrame,config:Dict):
        self.df = df
        self.config = config
        self.num_seqs = 500
        self.min_len = 18
        self.max_len = 30

    def get_random_seq(self):
        #create random sequences from bases: A,C,G,T with length 18-30
        random_seqs = []
        while len(random_seqs) < self.num_seqs:
            random_seq = ''.join(random.choices(['A','C','G','T'], k=randint(self.min_len,self.max_len)))
            if random_seq not in random_seqs and random_seq not in self.df.index:
                random_seqs.append(random_seq)

        return pd.DataFrame(index=random_seqs, data=['random']*len(random_seqs)\
            , columns =['Labels'])
    def get_augmented_df(self):
        random_df = self.get_random_seq()
        return random_df

class PrecursorAugmenter:
    def __init__(self,df:pd.DataFrame, config:Dict):
        self.df = df
        self.config = config
        self.mapping_dict = load(config['train_config'].mapping_dict_path)
        self.precursor_df = self.load_precursor_file()
        self.trained_on = config.trained_on
        
        self.min_num_samples_per_sc:int=1
        if self.trained_on == 'id':
            self.min_num_samples_per_sc = 8

        self.min_bin_size = 20
        self.max_bin_size = 30
        self.min_seq_len = 18
        self.max_seq_len = 30

    def load_precursor_file(self):
        try:
            precursor_df = pd.read_csv(self.config['train_config'].precursor_file_path, index_col=0)
            precursor_df.loc[:,'precursor_bins'] = (precursor_df.precursor_length/25).astype(int)
            return precursor_df
        except:
            print('Could not load precursor file')
            return None
        
    def compute_dynamic_bin_size(self,precursor_len:int, name:str=None) -> List[int]:
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

        if precursor_len < self.min_bin_size:
            return [precursor_len]
        else:
            precursor_bin_lens = split_precursor_to_bins(precursor_len,self.max_bin_size)
            reduced_len = self.max_bin_size-1
            while precursor_bin_lens[-1] < self.min_bin_size:
                precursor_bin_lens = split_precursor_to_bins(precursor_len,reduced_len)
                reduced_len -= 1
                if reduced_len < self.min_bin_size:
                    #add last two bins together
                    precursor_bin_lens[-2] += precursor_bin_lens[-1]
                    precursor_bin_lens = precursor_bin_lens[:-1]
                    break

            return precursor_bin_lens
        
    def get_bin_with_max_overlap(self,precursor_len:int,start_frag_pos:int,frag_len:int,name) -> int:
        '''
        This function returns the bin number of a fragment that overlaps the most with the fragment
        '''
        precursor_bin_lens = self.compute_dynamic_bin_size(precursor_len=precursor_len,name=name)
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
        return bin_no+1

    def get_precursor_info(self,mc:str,sc:str):
            
        xRNA_df = self.precursor_df.loc[self.precursor_df.sRNA_class == mc]
        xRNA_df.index = xRNA_df.index.str.replace('|','-', regex=False)
        prec_name = sc.split('_bin-')[0]
        bin_no = int(sc.split('_bin-')[1])
    
        if mc in ['snoRNA','lncRNA','protein_coding']:
            prec_name = mc+'-'+prec_name
            prec_row_df = xRNA_df.iloc[xRNA_df.index.str.contains(prec_name)]
            #check if prec_row_df is empty
            if prec_row_df.empty:
                xRNA_df = self.precursor_df.loc[self.precursor_df.sRNA_class == 'pseudo_'+mc]
                xRNA_df.index = xRNA_df.index.str.replace('|','-', regex=False)
                prec_row_df = xRNA_df.iloc[xRNA_df.index.str.contains(prec_name)]
                if prec_row_df.empty:
                    print(f'precursor {prec_name} not found in HBDxBase')
                    return pd.DataFrame()
    
            prec_row_df = prec_row_df.iloc[0]
        else:
            prec_row_df = xRNA_df.loc[f'{mc}-{prec_name}']
    
        precursor = prec_row_df.sequence
        return precursor,prec_name
    
    def populate_from_bin(self,sc:str,precursor:str,prec_name:str,existing_seqs:List[str]):
        '''
        This function will first get the bin no from the sc. 
        Then it will do three types of sampling:
        1. sample from the previous bin, insuring that the overlap with the middle bin is the highest
        2. sample from the next bin, insuring that the overlap with the middle bin is the highest
        3. sample from the middle bin, insuring that the overlap with the middle bin is the highest
        The staet idx should be the middle position of the previous bin, then start position is incremented until the end of the current bin
        '''
        bin_no = int(sc.split('_bin-')[1])
        bins = self.compute_dynamic_bin_size(len(precursor), prec_name)
        if len(bins) == 1:
            return pd.DataFrame()
        
        #bins start from 1 so should subtract 1
        bin_no -= 1

        #in case bin_no is 0
        try:
            previous_bin_start = sum(bins[:bin_no-1])
        except:
            previous_bin_start = 0
        middle_bin_start = sum(bins[:bin_no])
        next_bin_start = sum(bins[:bin_no+1])


        try:
            previous_bin_size = bins[bin_no-1]
        except:
            previous_bin_size = 0

        middle_bin_size = bins[bin_no]
        try: 
            next_bin_size = bins[bin_no+1]
        except:
            next_bin_size = 0


        start_idx = previous_bin_start + previous_bin_size//2 + 1 #+1 to make sure max overlap with prev bin is 14. max len/2 - 1
        sampled_seqs = []
        #increase start idx until the end of the current bin
        while start_idx < middle_bin_start+middle_bin_size:
            #compute the boundaries of the length of the fragment so that it would always overlap with the middle bin the most
            if start_idx < middle_bin_start:
                max_overlap_prev = middle_bin_start - start_idx
                end_idx = start_idx + randint(max(self.min_seq_len,max_overlap_prev*2+1),self.max_seq_len)
            else:# start_idx >= middle_bin_start:
                max_overlap_curr = next_bin_start - start_idx
                max_overlap_next = (start_idx + self.max_seq_len) - next_bin_start
                max_overlap_next = min(max_overlap_next,next_bin_size)
                if max_overlap_curr <= 9 or (max_overlap_next==0 and max_overlap_curr < self.min_seq_len):
                    end_idx = -1
                else:
                    end_idx = start_idx + randint(self.min_seq_len,min(self.max_seq_len,self.max_seq_len - max_overlap_next + max_overlap_curr - 1))
            #max overlap with the middle bin will never exceed half of min fragment (9) or,
            #  next bin size is 0 so frag will be shorter than 18
            if end_idx == -1:
                break

            tmp_seq = precursor[start_idx:end_idx]
            #introduce mismatches
            assert len(tmp_seq) >= self.min_seq_len and len(tmp_seq) <= self.max_seq_len, f'length of tmp_seq is {len(tmp_seq)}'
            if tmp_seq not in existing_seqs:
                sampled_seqs.append(tmp_seq)
            start_idx += 1
        
        #assertions
        for frag in sampled_seqs:
            all_occ = precursor.find(frag)
            if not isinstance(all_occ,list):
                all_occ = [all_occ]
            
            for occ in all_occ:
                curr_bin_no = self.get_bin_with_max_overlap(len(precursor),occ,len(frag),' ')
                # if curr_bin_no is different from bin_no+1 with more than 2 skip assertion
                if abs(curr_bin_no - (bin_no+1)) > 1:
                    continue
                assert curr_bin_no == bin_no+1, f'curr_bin_no is {curr_bin_no} and bin_no is {bin_no+1}'
            
        return pd.DataFrame(index=sampled_seqs, data=[sc]*len(sampled_seqs)\
            , columns =['Labels'])
    
    def populate_scs_with_bins(self):
        augmented_df = pd.DataFrame(columns=['Labels'])
        try:
            #append samples per sc for bin continuity
            unique_labels = self.df.Labels.value_counts()[self.df.Labels.value_counts() >= self.min_num_samples_per_sc].index.tolist()
            scs_list = []
            scs_before = []
            sc_after = []
            for sc in unique_labels:
                #retrieve_bin_from_precursor(other_sc_df,mapping_dict,sc)
                if type(sc) == str and '_bin-' in sc:
                    #get mc
                    try:
                        mc = self.mapping_dict[sc]
                    except:
                        sc_mc_mapper = lambda x: 'miRNA' if 'miR' in x else 'tRNA' if 'tRNA' in x else 'rRNA' if 'rRNA' in x else 'snRNA' if 'snRNA' in x else 'snoRNA' if 'snoRNA' in x else 'snoRNA' if 'SNO' in x else 'protein_coding' if 'RPL37A' in x else 'lncRNA' if 'SNHG1' in x else None
                        mc = sc_mc_mapper(sc)
                        if mc is None:
                            print(f'No mapping for {sc}')
                            continue
                    existing_seqs = self.df[self.df['Labels'] == sc].index
                    scs_list.append(sc)
                    scs_before.append(len(existing_seqs))
                    #augment fragments from prev or consecutive bin
                    precursor,prec_name = self.get_precursor_info(mc,sc)
                    sc2_df = self.populate_from_bin(sc,precursor,prec_name,existing_seqs)
                    augmented_df = augmented_df.append(sc2_df)
                    sc_after.append(len(sc2_df))
            #make a dict of scs and number of samples before and after augmentation
            scs_dict = {'sc':scs_list,'before':scs_before,'after':sc_after}
            scs_df = pd.DataFrame(scs_dict)
            scs_df.to_csv(f'scs_{self.trained_on}_df.csv')
        except:
            print('Could not sample from precursors')
        return augmented_df
    
    def get_augmented_df(self):
        return self.populate_scs_with_bins()
    
class DataAugmenter:
    '''
    This class sets the labels of the dataset to major class or sub class labels based on the clf_target
    major class: miRNA, tRNA ...
    sub class: mir-192-3p, rRNA-bin-30 ...
    Then if the models should be tained on ID models, it will augment the dataset with sequences sampled from the precursor file
    If the models should be trained on full, it will augment the dataset based on the following:
        1. Random sequences
        2. Recombined sequences
        3. Sequences sampled from the precursor file
        4. predictions of the sequences that previously had no annotation of low confidence but were predicted to be familiar by the ID models
    '''
    def __init__(self,df:pd.DataFrame, config:Dict):
        self.df = df
        self.config = config
        self.mapping_dict = load(config['train_config'].mapping_dict_path)
        self.trained_on = config.trained_on
        self.clf_target = config['model_config'].clf_target
        self.set_labels()

        self.precursor_augmenter = PrecursorAugmenter(self.df,self.config)
        self.random_augmenter = RandomSeqAugmenter(self.df,self.config)
        self.fusion_augmenter = RecombinedSeqAugmenter(self.df,self.config)
        self.id_model_augmenter = IDModelAugmenter(self.df,self.config)



    def set_labels(self):
        if 'hico' not in self.clf_target:
            self.df['Labels'] = self.df['subclass_name'].str.split(';', expand=True)[0]
        else:
            self.df['Labels'] = self.df['subclass_name'][self.df['hico'] == True]
    
    def convert_to_major_class_labels(self):
        if self.clf_target == 'major_class':
            self.df['Labels'] = self.df['Labels'].map(self.mapping_dict).astype('category')

  
    def combine_df(self,new_var_df:pd.DataFrame):
        #remove any sequences in augmented_df that exist in self.df.indexs
        duplicated_df = new_var_df[new_var_df.index.isin(self.df.index)]
        #log
        if len(duplicated_df):
            print(f'Number of duplicated sequences to be removed from ad: {duplicated_df.shape[0]}')

        new_var_df = new_var_df[~new_var_df.index.isin(self.df.index)].sample(frac=1)

        for col in self.df.columns:
            if col not in new_var_df.columns:
                new_var_df[col] = np.nan

        self.df = new_var_df.append(self.df)
        self.df.index = self.df.index.str.upper()  
        self.df.Labels = self.df.Labels.astype('category')
        return self.df

  
    def annotate_artificial_affix_seqs(self):
        #AA seqs are sequences that have 5' adapter
        aa_seqs = self.df[self.df['five_prime_adapter_filter'] == 0].index.tolist()
        self.df['Labels'] = self.df['Labels'].cat.add_categories('artificial_affix')
        self.df.loc[aa_seqs,'Labels'] = 'artificial_affix'


    
    def full_pipeline(self):
        self.df = self.id_model_augmenter.get_augmented_df()

    
    def post_augmentation(self):
        random_df = self.random_augmenter.get_augmented_df()
        df = self.precursor_augmenter.get_augmented_df()
        fusion_df = self.fusion_augmenter.get_augmented_df()
        df = df.append(fusion_df).append(random_df)
        self.df['Labels'] = self.df['Labels'].cat.add_categories({'random','recombined'})
        self.combine_df(df)

        self.convert_to_major_class_labels()
        self.annotate_artificial_affix_seqs()
        self.df['Labels'] = self.df['Labels'].cat.remove_unused_categories()
        self.df['Sequences'] = self.df.index.tolist()

        if 'struct' in self.config['model_config'].model_input:
            self.df['Secondary'] = fold_sequences(self.df.index.tolist(),temperature=37)[f'structure_37'].values

        return self.df

    def get_augmented_df(self):
        if self.trained_on == 'full':
            self.full_pipeline()
        return self.post_augmentation()