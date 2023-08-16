
import math
import warnings
from random import randint

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from ..utils import energy
from ..utils.utils import get_key
pd.options.mode.chained_assignment = None

class PrepareGeneData:
    def __init__(self, seqs_dot_bracket_labels: pd.DataFrame, config: DictConfig):
        self.seqs_dot_bracket_labels = seqs_dot_bracket_labels
        #shuffle
        self.seqs_dot_bracket_labels = self.seqs_dot_bracket_labels\
            .sample(frac=1)\
            .reset_index()
        #set both columns as objects
        self.seqs_dot_bracket_labels['Sequences'] = self.seqs_dot_bracket_labels['Sequences'].astype('object')
        self.seqs_dot_bracket_labels['Secondary'] = self.seqs_dot_bracket_labels['Secondary'].astype('object')

        # set max length to be <= 2 stds of distribtion of lengths
        #get max_length based on filtering
        self.get_outlier_length_threshold()
        

        #trimm all proteins to max length
        #34 samples removed
        self.limit_seqs_to_range()

        #get and set number of labels in config to be used later by the model
        model_config = config['model_config']
        model_config.num_classes = len(self.seqs_dot_bracket_labels['Labels'].unique())

        self.window = model_config.window
        self.seq_tokens_ids_dict = {}
        self.second_input_tokens_ids_dict = {}
        self.tokens_len = math.ceil(self.max_length / self.window)
        if model_config.tokenizer in ["overlap", "overlap_multi_window"]:
            self.tokens_len = self.max_length - (model_config.window - 1)
        self.tokenizer = model_config.tokenizer

    def get_outlier_length_threshold(self):
        lengths_arr = self.seqs_dot_bracket_labels['Sequences'].str.len()
        mean = np.mean(lengths_arr)
        standard_deviation = np.std(lengths_arr)
        distance_from_mean = abs(lengths_arr - mean)
        in_distribution = distance_from_mean < 2 * standard_deviation

        inlier_lengths = np.sort(lengths_arr[in_distribution].unique())
        self.max_length = np.max(inlier_lengths)
        self.min_length = np.min(inlier_lengths)
        return 
    

    def limit_seqs_to_range(self):
        '''
        Trimms seqs longer than maximum len and deletes seqs shorter than min length
        '''
        min_to_be_deleted = []
        for idx,seq in enumerate(self.seqs_dot_bracket_labels['Sequences']):
            if len(seq) > self.max_length:
                self.seqs_dot_bracket_labels['Sequences'].iloc[idx] = \
                    self.seqs_dot_bracket_labels['Sequences'].iloc[idx][:self.max_length]
                self.seqs_dot_bracket_labels['Secondary'].iloc[idx] = \
                    self.seqs_dot_bracket_labels['Secondary'].iloc[idx][:self.max_length]
            elif len(seq) < self.min_length:
                #deleted sequence indices
                min_to_be_deleted.append(idx)
        #delete min sequences
        if len(min_to_be_deleted):
            self.seqs_dot_bracket_labels = self.seqs_dot_bracket_labels\
                .drop(min_to_be_deleted)\
                .reset_index()
    
    def get_secondary_structure(self,sequences):
        secondary = energy.fold_sequences(sequences.tolist())
        return secondary['structure_37'].values

 

    # function generating non overlapping tokens of a feature sample
    def chunkstring(self, string, window):
        return (string[0 + i : window + i] for i in range(0, len(string), window))

    # function generating non overlapping tokens of a feature sample
    def chunkstring_overlap(self, string, window):
        return (
            string[0 + i : window + i] for i in range(0, len(string) - window + 1, 1)
        )

    def tokenize_samples(self, window,sequences_to_be_tokenized,inference:bool=False) -> np.ndarray:
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
        if self.tokenizer in ["overlap", "overlap_multi_window"]:
            feature_tokens_gen = list(
                self.chunkstring_overlap(feature, window)
                for feature in sequences_to_be_tokenized
            )
        elif self.tokenizer == "no_overlap":
            feature_tokens_gen = list(
                self.chunkstring(feature, window) for feature in sequences_to_be_tokenized
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
                        warnings.warn(f"The sequence token: {token} was not seen previously by the model. Token will be replaced by a random token")
                        id = randint(1,len(self.seq_tokens_ids_dict.keys()) - 1)
                        token = get_key(self.seq_tokens_ids_dict,id)
                # append id of token
                sample_token_id.append(self.seq_tokens_ids_dict[token])

            # append ids of tokenized sample
            sample_token_ids.append(np.array(sample_token_id))

            sample_token = np.array(sample_token)
            samples_tokenized.append(sample_token)
        # save vocab
        return (np.array(samples_tokenized), np.array(sample_token_ids))

    def tokenize_secondary_structure(self, window,sequences_to_be_tokenized,inference:bool=False) -> np.ndarray:
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
        if self.tokenizer in ["overlap", "overlap_multi_window"]:
            feature_tokens_gen = list(
                self.chunkstring_overlap(feature, window)
                for feature in sequences_to_be_tokenized
            )
        elif self.tokenizer == "no_overlap":
            feature_tokens_gen = list(
                self.chunkstring(feature, window) for feature in sequences_to_be_tokenized
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
                        token = get_key(self.second_input_tokens_ids_dict,id)
                # append id of token
                sample_token_id.append(self.second_input_tokens_ids_dict[token])
            # append ids of tokenized sample
            sample_token_ids.append(sample_token_id)
            samples_tokenized.append(sample_token)
            
        #append pads 
        #max length is number of different temp used* max token len PLUS the concat token 
        # between two secondary structures represented at two diff temperatures
        self.second_input_token_len = self.tokens_len
        seqs_length = [] 
        for seq_idx, token in enumerate(sample_token_ids):
            sample_len = len(token)
            seqs_length.append(sample_len)
            sample_token_ids[seq_idx].extend(
                [self.second_input_tokens_ids_dict["pad"] for _ in range(int(self.second_input_token_len - sample_len))]
            )
            samples_tokenized[seq_idx].extend(
                ["pad" for _ in range(int(self.second_input_token_len - sample_len))]
            )
            sample_token_ids[seq_idx] = np.array(sample_token_ids[seq_idx])
            samples_tokenized[seq_idx] = np.array(samples_tokenized[seq_idx])
        # save vocab
        return (samples_tokenized, sample_token_ids,seqs_length)

    def tokenize_samples_wrapper(self,sequences_to_be_tokenized,tokenize_func,ss_flag = False,inference:bool=False):
        if ss_flag:
            samples_tokenized, sample_token_ids,seqs_length = tokenize_func(self.window,sequences_to_be_tokenized,inference)
        else:
            samples_tokenized, sample_token_ids = tokenize_func(self.window,sequences_to_be_tokenized,inference)
        if ss_flag:
            return samples_tokenized,sample_token_ids,seqs_length
        return samples_tokenized,sample_token_ids
        
    def get_preprocessed_data_df(self,inference:bool=False):
        #tokenize sequences and secondary structures for all sequences 

        samples_tokenized,sample_token_ids = self.tokenize_samples_wrapper(self.seqs_dot_bracket_labels['Sequences'],
                                                tokenize_func=self.tokenize_samples,inference=inference)
        _,sec_str_token_ids,seqs_length = self.tokenize_samples_wrapper(self.seqs_dot_bracket_labels['Secondary'],
                                                self.tokenize_secondary_structure,ss_flag =True,inference=inference)


        labels = self.seqs_dot_bracket_labels['Labels'].values

        #create labels_df
        iterables = [["Labels"], np.arange(1, dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        labels_df = pd.DataFrame(columns=index, data=labels)

        # create pandas dataframe  for token ids of sequences
        iterables = [["tokens_id"], np.arange(self.tokens_len, dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        tokens_id_df = pd.DataFrame(columns=index, data=sample_token_ids)

        # create pandas dataframe  for tokens of sequences
        iterables = [["tokens"], np.arange(self.tokens_len, dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        tokens_df = pd.DataFrame(columns=index, data=samples_tokenized)

        # create pandas dataframe  token ids of secondary structures
        iterables = [["second_input"], np.arange(self.second_input_token_len,dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        ss_tokens_id_df = pd.DataFrame(columns=index, data=sec_str_token_ids)

        #create seqs length, will be later used by model
        # create pandas dataframe  token ids of secondary structures
        iterables = [["seqs_length"], np.arange(1,dtype=int)]
        index = pd.MultiIndex.from_product(iterables, names=["type of data", "indices"])
        seqs_length_df = pd.DataFrame(columns=index, data=seqs_length)
        all_df = labels_df.join(tokens_df).join(tokens_id_df).join(ss_tokens_id_df).join(seqs_length_df)
        if inference:
            return all_df
        else:
            return all_df.sample(frac=1)