import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import (compute_class_weight,
                                        compute_sample_weight)
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from ..utils.energy import fold_sequences
from ..utils.file import load, save
from ..utils.utils import (revert_seq_tokenization,
                           update_config_with_dataset_params_benchmark,
                           update_config_with_dataset_params_tcga)

logger = logging.getLogger(__name__)
class PrepareGeneData:
    def __init__(self,tokenizer,configs):
        self.tokenizer = tokenizer
        self.configs = configs
        self.seed = configs.seed
        self.trained_on = configs.trained_on
        self.device = configs["train_config"].device
        self.splits_df_dict = {}
        self.min_num_samples_per_class = 10

    def convert_to_tensor(self,in_arr,convert_type):
        tensor_dtype = torch.long if convert_type == int else torch.float
        return torch.tensor(
            np.array(in_arr, dtype=convert_type),
            dtype=tensor_dtype,
        ).to(device=self.device)
    
    def get_features_per_split(self):
        model_input_cols = ['tokens_id','second_input','seqs_length']
        features_dict = {}
        for split_df in self.splits_df_dict.keys():
            split_data = self.convert_to_tensor(self.splits_df_dict[split_df][model_input_cols].values,convert_type=float)
            split = '_'.join(split_df.split('_')[:-1])
            features_dict[f'{split}_data'] = split_data

        return features_dict
    
    def append_sample_weights(self,splits_features_dict):

        for split_df in self.splits_df_dict.keys():
            if split_df in ['train_df','valid_df','test_df']:
                split_weights =  self.convert_to_tensor(compute_sample_weight('balanced',self.splits_df_dict[split_df]['Labels'][0]),convert_type=float)
            else:
                split_weights = self.convert_to_tensor(np.ones(self.splits_df_dict[split_df].shape[0]),convert_type=float)
            split = '_'.join(split_df.split('_')[:-1])
            splits_features_dict[f'{split}_data'] = torch.cat([splits_features_dict[f'{split}_data'],split_weights[:,None]],dim=1)   

        return

    def get_labels_per_split(self):
        #encode labels 
        enc = LabelEncoder()
        enc.fit(self.splits_df_dict["train_df"]['Labels'])
        #save mapping dict to config
        self.configs["model_config"].class_mappings = enc.classes_.tolist()

        labels_dict = {}
        labels_numeric_dict = {}
        for split_df in self.splits_df_dict.keys():
            split = '_'.join(split_df.split('_')[:-1])

            split_labels = self.splits_df_dict[split_df]['Labels']
            if split_df in ['train_df','valid_df','test_df']:
                split_labels_numeric = self.convert_to_tensor(enc.transform(split_labels), convert_type=int)
            else:
                split_labels_numeric = self.convert_to_tensor(np.zeros((split_labels.shape[0])), convert_type=int)
            
            labels_dict[f'{split}_labels'] = split_labels
            labels_numeric_dict[f'{split}_labels_numeric'] = split_labels_numeric
        
        #compute class weight
        class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(labels_dict['train_labels']),y=labels_dict['train_labels'][0].values)
        
        #omegaconfig does not support float64 as datatype so conversion to str is done 
        # and reconversion is done in criterion
        self.configs['model_config'].class_weights = [str(x) for x in list(class_weights)]


        return labels_dict | labels_numeric_dict
    

    def get_seqs_per_split(self):
        rna_seq_dict = {}
        for split_df in self.splits_df_dict.keys():
            split = '_'.join(split_df.split('_')[:-1])
            rna_seq_dict[f'{split}_rna_seq'] = revert_seq_tokenization(self.splits_df_dict[split_df]["tokens"],self.configs)

        return rna_seq_dict

    def duplicate_fewer_classes(self,df):
        #get quantity of each class and append it as a column
        df["Quantity",'0'] = df["Labels"].groupby([0])[0].transform("count")
        frequent_samples_df = df[df["Quantity",'0'] >= self.min_num_samples_per_class].reset_index(drop=True)
        fewer_samples_df = df[df["Quantity",'0'] < self.min_num_samples_per_class].reset_index(drop=True)
        unique_fewer_samples_df = fewer_samples_df.drop_duplicates(subset=[('Labels',0)], keep="last")
        unique_fewer_samples_df['Quantity','0'] -= self.min_num_samples_per_class
        unique_fewer_samples_df['Quantity','0'] = unique_fewer_samples_df['Quantity','0'].abs()
        repeated_fewer_samples_df = unique_fewer_samples_df.loc[unique_fewer_samples_df.index.repeat(unique_fewer_samples_df.Quantity['0'])]
        repeated_fewer_samples_df = repeated_fewer_samples_df.reset_index(drop=True)
        df = frequent_samples_df.append(repeated_fewer_samples_df).append(fewer_samples_df).reset_index(drop=True)
        df.drop(columns = ['Quantity'],inplace=True)
        return df

    def remove_fewer_samples(self,data_df):
        counts = data_df['Labels'].value_counts()
        fewer_class_ids = counts[counts < self.min_num_samples_per_class].index
        fewer_class_labels = [i[0]  for i in fewer_class_ids]
        fewer_samples_per_class_df = data_df.loc[data_df['Labels'].isin(fewer_class_labels).values, :]
        fewer_ids = data_df.index.isin(fewer_samples_per_class_df.index)
        data_df = data_df[~fewer_ids]
        return fewer_samples_per_class_df,data_df

    def split_tcga(self,data_df):
        #remove artificial_affix
        artificial_df = data_df.loc[data_df['Labels'][0].isin(['random','recombined','artificial_affix'])]
        art_ids = data_df.index.isin(artificial_df.index)
        data_df = data_df[~art_ids]
        data_df = data_df.reset_index(drop=True)

        #remove no annotations
        no_annotaton_df = data_df.loc[data_df['Labels'].isnull().values]
        n_a_ids = data_df.index.isin(no_annotaton_df.index)
        data_df = data_df[~n_a_ids].reset_index(drop=True)
        no_annotaton_df = no_annotaton_df.reset_index(drop=True)

        if self.trained_on == 'full':
            data_df = self.duplicate_fewer_classes(data_df)
            ood_dict = {}
        else:
            ood_df,data_df = self.remove_fewer_samples(data_df)
            ood_dict = {"ood_df":ood_df}
        #split data
        train_df,valid_test_df = train_test_split(data_df,stratify=data_df["Labels"],train_size=0.8,random_state=self.seed)
        if self.trained_on == 'id':
            valid_df,test_df = train_test_split(valid_test_df,stratify=valid_test_df["Labels"],train_size=0.5,random_state=self.seed)
        else:
            #we need to use all n sequences in the training set, however, unseen samples should be gathered for training novelty prediction,
            #otherwise NLD for test would be zero
            #remove one sample from each class to test_df
            test_df = valid_test_df.drop_duplicates(subset=[('Labels',0)], keep="last")
            test_ids = valid_test_df.index.isin(test_df.index)
            valid_df = valid_test_df[~test_ids].reset_index(drop=True)
            train_df = train_df.append(valid_df).reset_index(drop=True)

        self.splits_df_dict =  {"train_df":train_df,"valid_df":valid_df,"test_df":test_df,"artificial_df":artificial_df,"no_annotation_df":no_annotaton_df} | ood_dict

    def prepare_data_tcga(self):
        """
        This function recieves tokenizer and prepares the data in a format suitable for training
        It also set default parameters in the config that cannot be known until preprocessing step
        is done.
        """
        all_data_df = self.tokenizer.get_tokenized_data()

        #split data 
        self.split_tcga(all_data_df)

        num_samples = self.splits_df_dict['train_df'].shape[0]
        num_classes = len(self.splits_df_dict['train_df'].Labels.value_counts()[self.splits_df_dict['train_df'].Labels.value_counts()>0])
        #log
        logger.info(f'Training with {num_classes} classes and {num_samples} samples')

        #get features, labels, and seqs per split
        splits_features_dict = self.get_features_per_split()
        self.append_sample_weights(splits_features_dict)
        splits_labels_dict = self.get_labels_per_split()
        splits_seqs_dict = self.get_seqs_per_split()

        
        #prepare validation set for skorch
        valid_ds = Dataset(splits_features_dict["valid_data"],splits_labels_dict["valid_labels_numeric"])
        valid_ds = predefined_split(valid_ds)

        #combine all dicts
        all_data = splits_features_dict | splits_labels_dict | splits_seqs_dict | \
            {"valid_ds":valid_ds}

        ###update self.configs
        update_config_with_dataset_params_tcga(self.tokenizer,all_data_df,self.configs)
        self.configs["model_config"].num_classes = len(all_data['train_labels'][0].unique())
        self.configs["train_config"].batch_per_epoch = int(all_data["train_data"].shape[0]\
            /self.configs["train_config"].batch_size)

        return all_data

    
    

