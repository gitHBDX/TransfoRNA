
import logging
import math
import os
import warnings
from random import randint

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from omegaconf import DictConfig, open_dict

from ..utils import energy
from ..utils.file import save

logger = logging.getLogger(__name__)

class SeqTokenizer:
    '''
    This class should contain functions that other data specific classes should inherit from.
    '''
    def __init__(self,seqs_dot_bracket_labels: pd.DataFrame, config: DictConfig):

        self.seqs_dot_bracket_labels = seqs_dot_bracket_labels
        #shuffle
        self.seqs_dot_bracket_labels = self.seqs_dot_bracket_labels\
            .sample(frac=1)\
            .reset_index(drop=True)
        
        #get input of model
        self.model_input = config["model_config"].model_input


        # set max length to be <= 2 stds of distribtion of lengths
        if config["train_config"].filter_seq_length:
            self.get_outlier_length_threshold()
            self.limit_seqs_to_range()

        else:
            self.max_length = self.seqs_dot_bracket_labels['Sequences'].str.len().max()
            self.min_length = 0

        with open_dict(config):
            config["model_config"]["max_length"] = np.int64(self.max_length).item()
            config["model_config"]["min_length"] = np.int64(self.min_length).item()
        
        self.window = config["model_config"].window    
        self.tokens_len = math.ceil(self.max_length / self.window)
        if config["model_config"].tokenizer in ["overlap", "overlap_multi_window"]:
            self.tokens_len = int(self.max_length - (config["model_config"].window - 1))
        self.tokenizer = config["model_config"].tokenizer
        

        self.seq_len_dist = self.seqs_dot_bracket_labels['Sequences'].str.len().value_counts()
        #init tokens dict
        self.seq_tokens_ids_dict = {}
        self.second_input_tokens_ids_dict = {}

        #get and set number of labels in config to be used later by the model
        config["model_config"].num_classes = len(self.seqs_dot_bracket_labels['Labels'].unique())

        self.set_class_attr()


    def get_outlier_length_threshold(self):
        lengths_arr = self.seqs_dot_bracket_labels['Sequences'].str.len()
        mean = np.mean(lengths_arr)
        standard_deviation = np.std(lengths_arr)
        distance_from_mean = abs(lengths_arr - mean)
        in_distribution = distance_from_mean < 2 * standard_deviation

        inlier_lengths = np.sort(lengths_arr[in_distribution].unique())
        self.max_length = int(np.max(inlier_lengths))
        self.min_length = int(np.min(inlier_lengths))
        logger.info(f'maximum and minimum sequence length is set to: {self.max_length} and {self.min_length}')
        return 
    

    def limit_seqs_to_range(self):
        '''
        Trimms seqs longer than maximum len and deletes seqs shorter than min length
        '''
        df = self.seqs_dot_bracket_labels
        min_to_be_deleted = []

        num_longer_seqs = sum(df['Sequences'].str.len()>self.max_length)
        if num_longer_seqs:
            #raise ValueError(f"Number of sequences longer than max length: {num_longer_seqs}")
            print(f"Number of sequences to be trimmed: {num_longer_seqs}")


        for idx,seq in enumerate(df['Sequences']):
            if len(seq) > self.max_length:
                df['Sequences'].iloc[idx] = \
                    df['Sequences'].iloc[idx][:self.max_length]
                
            elif len(seq) < self.min_length:
                #deleted sequence indices
                min_to_be_deleted.append(str(idx))
        #delete min sequences
        if len(min_to_be_deleted):
            df = df.drop(min_to_be_deleted).reset_index(drop=True)
            logger.info(f"Number of sequences shroter sequences to be removed: {len(min_to_be_deleted)}")
        self.seqs_dot_bracket_labels = df
    
    def get_secondary_structure(self,sequences):
        secondary = energy.fold_sequences(sequences.tolist())
        return secondary['structure_37'].values
    
    # function generating non overlapping tokens of a feature sample
    def chunkstring_overlap(self, string, window):
        return (
            string[0 + i : window + i] for i in range(0, len(string) - window + 1, 1)
        )
    # function generating non overlapping tokens of a feature sample
    def chunkstring_no_overlap(self, string, window):
        return (string[0 + i : window + i] for i in range(0, len(string), window))
    

    def tokenize_samples(self, window:int,sequences_to_be_tokenized:pd.DataFrame,inference:bool=False,tokenizer:str="overlap") -> np.ndarray:
        """
        This function tokenizes rnas based on window(window)
        with or without overlap according to the current tokenizer option.
        In case of overlap:
        example: Token :AACTAGA, window: 3
        output: AAC,ACT,CTA,TAG,AGA

        In case no_overlap:
        example: Token :AACTAGA, window: 3
        output: AAC,TAG,A
        """
        # get feature tokens
        if "overlap" in tokenizer:
            feature_tokens_gen = list(
                self.chunkstring_overlap(feature, window)
                for feature in sequences_to_be_tokenized
            )
        elif tokenizer == "no_overlap":
            feature_tokens_gen = list(
                self.chunkstring_no_overlap(feature, window) for feature in sequences_to_be_tokenized
            )
        # get sample tokens and pad them
        samples_tokenized = []
        sample_token_ids = []
        if not self.seq_tokens_ids_dict:
            self.seq_tokens_ids_dict = {"pad": 0}
            
        for gen in feature_tokens_gen:
            sample_token_id = []
            sample_token = list(gen)
            sample_len = len(sample_token)
            # append paddings
            sample_token.extend(
                ["pad" for _ in range(int(self.tokens_len - sample_len))]
            )
            # convert tokens to ids
            for token in sample_token:
                # if token doesnt exist in dict, create one
                if token not in self.seq_tokens_ids_dict:
                    if not inference:
                        id = len(self.seq_tokens_ids_dict.keys())
                        self.seq_tokens_ids_dict[token] = id
                    else:
                        #if new token found during inference, then select random token (considered as noise)
                        logger.warning(f"The sequence token: {token} was not seen previously by the model. Token will be replaced by a random token")
                        id = randint(1,len(self.seq_tokens_ids_dict.keys()) - 1)
                        token = self.seq_tokens_ids_dict[id]
                # append id of token
                sample_token_id.append(self.seq_tokens_ids_dict[token])

            # append ids of tokenized sample
            sample_token_ids.append(np.array(sample_token_id))

            sample_token = np.array(sample_token)
            samples_tokenized.append(sample_token)
        # save vocab
        return (np.array(samples_tokenized), np.array(sample_token_ids))
    
    def tokenize_secondary_structure(self, window,sequences_to_be_tokenized,inference:bool=False,tokenizer= "overlap") -> np.ndarray:
        """
        This function tokenizes rnas based on window(window)
        with or without overlap according to the current tokenizer option.
        In case of overlap:
        example: Token :...()..., window: 3
        output: ...,..(,.(),().,)..,...

        In case no_overlap:
        example: Token :...()..., window: 3
        output: ...,().,..
        """
        samples_tokenized = []
        sample_token_ids = []
        if not self.second_input_tokens_ids_dict:
            self.second_input_tokens_ids_dict = {"pad": 0}

        # get feature tokens
        if "overlap" in tokenizer:
            feature_tokens_gen = list(
                self.chunkstring_overlap(feature, window)
                for feature in sequences_to_be_tokenized
            )
        elif "no_overlap" == tokenizer:
            feature_tokens_gen = list(
                self.chunkstring_no_overlap(feature, window) for feature in sequences_to_be_tokenized
            )
        # get sample tokens and pad them
        for seq_idx, gen in enumerate(feature_tokens_gen):
            sample_token_id = []
            sample_token = list(gen)
            
            # convert tokens to ids
            for token in sample_token:
                # if token doesnt exist in dict, create one
                if token not in self.second_input_tokens_ids_dict:
                    if not inference:
                        id = len(self.second_input_tokens_ids_dict.keys())
                        self.second_input_tokens_ids_dict[token] = id
                    else:
                        #if new token found during inference, then select random token (considered as noise)
                        warnings.warn(f"The secondary structure token: {token} was not seen previously by the model. Token will be replaced by a random token")
                        id = randint(1,len(self.second_input_tokens_ids_dict.keys()) - 1)
                        token = self.second_input_tokens_ids_dict[id]
                # append id of token
                sample_token_id.append(self.second_input_tokens_ids_dict[token])
            # append ids of tokenized sample
            sample_token_ids.append(sample_token_id)
            samples_tokenized.append(sample_token)
            
        #append pads 
        #max length is number of different temp used* max token len PLUS the concat token 
        # between two secondary structures represented at two diff temperatures
        self.second_input_token_len = self.tokens_len
        for seq_idx, token in enumerate(sample_token_ids):
            sample_len = len(token)
            sample_token_ids[seq_idx].extend(
                [self.second_input_tokens_ids_dict["pad"] for _ in range(int(self.second_input_token_len - sample_len))]
            )
            samples_tokenized[seq_idx].extend(
                ["pad" for _ in range(int(self.second_input_token_len - sample_len))]
            )
            sample_token_ids[seq_idx] = np.array(sample_token_ids[seq_idx])
            samples_tokenized[seq_idx] = np.array(samples_tokenized[seq_idx])
        # save vocab
        return (samples_tokenized, sample_token_ids)
    
    def set_class_attr(self):
        #set seq,struct and exp and labels
        self.seq = self.seqs_dot_bracket_labels["Sequences"]
        if 'struct' in self.model_input:
            self.struct = self.seqs_dot_bracket_labels["Secondary"]

        self.labels = self.seqs_dot_bracket_labels['Labels']

    def prepare_multi_idx_pd(self,num_coln,pd_name,pd_value):
        iterables = [[pd_name], np.arange(num_coln)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        return pd.DataFrame(columns=index, data=pd_value)

    def phase_sequence(self,sample_token_ids):
        phase0 = sample_token_ids[:,::2]
        phase1 = sample_token_ids[:,1::2]
        #in case max_length is an odd number phase 0 will be 1 entry larger than phase 1 @ dim=1 
        if phase0.shape!= phase1.shape:
            phase1 = np.concatenate([phase1,np.zeros(phase1.shape[0])[...,np.newaxis]],axis=1)
        sample_token_ids = phase0
        
        return sample_token_ids,phase1

    def custom_roll(self,arr, n_shifts_per_row):
        '''
        shifts each row of a numpy array according to n_shifts_per_row
        '''
        m = np.asarray(n_shifts_per_row)
        arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy() #need `copy`
        strd_0, strd_1 = arr_roll.strides
        n = arr.shape[1]
        result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))

        return result[np.arange(arr.shape[0]), (n-m)%n]

    def save_token_dicts(self):
        #save token dicts
        save(data = self.second_input_tokens_ids_dict,path = os.getcwd()+'/second_input_tokens_ids_dict')
        save(data = self.seq_tokens_ids_dict,path = os.getcwd()+'/seq_tokens_ids_dict')
        #save token dicts
        save(data = self.second_input_tokens_ids_dict,path = os.getcwd()+'/second_input_tokens_ids_dict')
        save(data = self.seq_tokens_ids_dict,path = os.getcwd()+'/seq_tokens_ids_dict')

    def get_tokenized_data(self,inference:bool=False):
        #tokenize sequences
        samples_tokenized,sample_token_ids = self.tokenize_samples(self.window,self.seq,inference)

        print(f'Vocab size for primary sequences: {len(self.seq_tokens_ids_dict.keys())}')
        print(f'Vocab size for secondary structure: {len(self.second_input_tokens_ids_dict.keys())}')
        print(f'Number of sequences used for tokenization: {samples_tokenized.shape[0]}')

        #tokenize struct if used
        if "comp" in self.model_input:
            #get compliment of self.seq
            self.seq_comp = []
            for feature in self.seq:
                feature  = feature.replace('A','%temp%').replace('T','A')\
                                  .replace('C','%temp2%').replace('G','C')\
                                  .replace('%temp%','T').replace('%temp2%','G')
                self.seq_comp.append(feature)
            #store seq_tokens_ids_dict
            self.seq_tokens_ids_dict_temp = self.seq_tokens_ids_dict
            self.seq_tokens_ids_dict = None
            #tokenize compliment
            _,seq_comp_str_token_ids = self.tokenize_samples(self.window,self.seq_comp,inference)
            sec_input_value = seq_comp_str_token_ids
            #store second input seq_tokens_ids_dict
            self.second_input_tokens_ids_dict = self.seq_tokens_ids_dict
            self.seq_tokens_ids_dict = self.seq_tokens_ids_dict_temp


        #tokenize struct if used
        if "struct" in self.model_input:
            _,sec_str_token_ids = self.tokenize_secondary_structure(self.window,self.struct,inference)
            sec_input_value = sec_str_token_ids


        #add seq-seq if used
        if "seq-seq" in self.model_input:
            sample_token_ids,sec_input_value = self.phase_sequence(sample_token_ids)
            self.second_input_tokens_ids_dict = self.seq_tokens_ids_dict

        #in case of baseline or only "seq", the second input is dummy
        #TODO:: refactor transforna to accept models with a single input (baseline and "seq") 
        # without occupying unnecessary resources
        if "seq-rev" in self.model_input or "baseline" in self.model_input or self.model_input == 'seq':
            sample_token_ids_rev = sample_token_ids[:,::-1]
            n_zeros = np.count_nonzero(sample_token_ids_rev==0, axis=1)
            sec_input_value = self.custom_roll(sample_token_ids_rev, -n_zeros)
            self.second_input_tokens_ids_dict = self.seq_tokens_ids_dict




        seqs_length = list(sum(sample_token_ids.T !=0))

        labels_df = self.prepare_multi_idx_pd(1,"Labels",self.labels.values)
        tokens_id_df = self.prepare_multi_idx_pd(sample_token_ids.shape[1],"tokens_id",sample_token_ids)
        tokens_df = self.prepare_multi_idx_pd(samples_tokenized.shape[1],"tokens",samples_tokenized)
        sec_input_df = self.prepare_multi_idx_pd(sec_input_value.shape[1],'second_input',sec_input_value)
        seqs_length_df = self.prepare_multi_idx_pd(1,"seqs_length",seqs_length)

        all_df = labels_df.join(tokens_df).join(tokens_id_df).join(sec_input_df).join(seqs_length_df)

        #save token dicts
        self.save_token_dicts()
        return all_df