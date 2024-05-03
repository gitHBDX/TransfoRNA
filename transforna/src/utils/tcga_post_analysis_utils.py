import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from .file import create_dirs, load

logger = logging.getLogger(__name__)
class Results_Handler():
    def __init__(self,embedds_path:str,splits:List,mc_flag:bool=False,read_dataset:bool=False,create_knn_graph:bool=False,run_name:str=None,save_results:bool=False) -> None:
        self.save_results = save_results
        self.all_splits = ['train','valid','test','ood','artificial','no_annotation']
        if splits == ['all']:
            self.splits = self.all_splits
        else:
            self.splits = splits
        self.mc_flag = mc_flag

        #if embedds is not at the end of the embedds_path then append it
        if not embedds_path.endswith('embedds'):
            embedds_path = embedds_path+'/embedds'

        _,self.splits_df_dict = self.get_data(embedds_path,self.splits)

        #set column names
        self.embedds_cols:List = [col for col in self.splits_df_dict[f'{splits[0]}_df'] if "Embedds" in col[0]]
        self.seq_col:str = 'RNA Sequences'
        self.label_col:str = 'Labels'

        #create directories
        self.parent_path:str = '/'.join(embedds_path.split('/')[:-1])
        self.figures_path:str = self.parent_path+'/figures'
        self.analysis_path:str = self.parent_path+'/analysis'
        self.meta_path:str = self.parent_path+'/meta'
        self.umaps_path:str = self.parent_path+'/umaps'
        self.post_models_path:str = self.parent_path+'/post_models'
        create_dirs([self.figures_path,self.analysis_path,self.post_models_path])

        #get half of embedds cols if the model is Seq
        model_name = self.get_hp_param(hp_param="model_name")
        if model_name == 'seq':
            self.embedds_cols = self.embedds_cols[:len(self.embedds_cols)//2]

        if not run_name:
            self.run_name = self.get_hp_param(hp_param="model_input")
            if type(self.run_name) == list:
                self.run_name = '-'.join(self.run_name)

        ad_path = self.get_hp_param(hp_param="dataset_path_train")
        if read_dataset:
            self.dataset = load(ad_path)
            if isinstance(self.dataset,AnnData):
                self.dataset = self.dataset.var
        

        self.seperate_label_from_split(split='artificial',removed_label='artificial_affix')
        self.seperate_label_from_split(split='artificial',removed_label='random')
        self.seperate_label_from_split(split='artificial',removed_label='recombined')

        self.sc_to_mc_mapper_dict = self.load_mc_mapping_dict()

        #get whether curr results are trained on ID or FULL
        self.trained_on = self.get_hp_param(hp_param="trained_on")
        #the main config of models trained on ID is not logged as for FULL
        if self.trained_on == None:
            self.trained_on = 'id'


        #read train to be used for knn training and inference
        train_df = self.splits_df_dict['train_df']
        
        self.knn_seqs = train_df[self.seq_col].values
        self.knn_labels = train_df[self.label_col].values

        #create knn model if does not exist
        if create_knn_graph:
            self.create_knn_model()

    def create_knn_model(self):
        #get all train embedds
        train_embedds = self.splits_df_dict['train_df'][self.embedds_cols].values
        #linalg
        train_embedds = train_embedds/np.linalg.norm(train_embedds,axis=1)[:,None]
        #create knn model
        self.knn_model = NearestNeighbors(n_neighbors=10,algorithm='brute',n_jobs=-1)
        self.knn_model.fit(train_embedds)
        #save knn model
        filename = self.post_models_path+'/knn_model.sav'
        pickle.dump(self.knn_model,open(filename,'wb'))
        return
    
    def get_knn_model(self):
        filename = self.post_models_path+'/knn_model.sav'
        self.knn_model = pickle.load(open(filename,'rb'))
        return

    def seperate_label_from_split(self,split,removed_label:str='artificial_affix'):
        
        if split in self.splits:
            logger.debug(f"splitting {removed_label} from split: {split}")


            #get art affx
            removed_label_df = self.splits_df_dict[f"{split}_df"].loc[self.splits_df_dict[f"{split}_df"][self.label_col]['0'] == removed_label]

            #append art affx as key
            self.splits_df_dict[f'{removed_label}_df'] = removed_label_df
            #remove art affx from ood
            removed_label_ids = self.splits_df_dict[f"{split}_df"].index.isin(removed_label_df.index)
            self.splits_df_dict[f"{split}_df"] = self.splits_df_dict[f"{split}_df"][~removed_label_ids].reset_index(drop=True)

            #resetf {split}_affix_idx
            self.splits_df_dict[f'{removed_label}_df'] = self.splits_df_dict[f'{removed_label}_df'].reset_index(drop=True)
            self.all_splits.append(f'{removed_label}')


    def append_loco_variants(self):
        train_classes = self.splits_df_dict["train_df"]["Logits"].columns.values
        if self.mc_flag:
            all_loco_classes_df = self.dataset['small_RNA_class_annotation'][self.dataset['small_RNA_class_annotation_hico'].isnull()].str.split(';', expand=True)
        else:
            all_loco_classes_df = self.dataset['subclass_name'][self.dataset['hico'].isnull()].str.split(';', expand=True)

        all_loco_classes = all_loco_classes_df.values

        #TODO: optimize getting unique values 
        loco_classes = []
        for col in all_loco_classes_df.columns:
            loco_classes.extend(all_loco_classes_df[col].unique())

        loco_classes = list(set(loco_classes))
        if np.nan in loco_classes:
            loco_classes.remove(np.nan)
        if None in loco_classes:
            loco_classes.remove(None)

        #compute loco not in train mask
        loco_classes_not_in_train = list(set(loco_classes).difference(set(train_classes)))
        loco_mask_not_in_train_df = all_loco_classes_df.isin(loco_classes_not_in_train)


        mixed_and_not_in_train_df = all_loco_classes_df.iloc[loco_mask_not_in_train_df.values.sum(axis=1) >= 1]
        train_classes_mask = mixed_and_not_in_train_df.isin(train_classes)

        loco_not_in_train_df  = mixed_and_not_in_train_df[train_classes_mask.values.sum(axis=1) == 0]
        loco_mixed_df = mixed_and_not_in_train_df[~(train_classes_mask.values.sum(axis=1) == 0)]

        nans_and_loco_train_df = all_loco_classes_df.iloc[loco_mask_not_in_train_df.values.sum(axis=1) == 0]
        nans_mask = nans_and_loco_train_df.isin([None,np.nan])
        nanas_df = nans_and_loco_train_df[nans_mask.values.sum(axis=1) == len(nans_mask.columns)]
        loco_in_train_df = nans_and_loco_train_df[nans_mask.values.sum(axis=1) < len(nans_mask.columns)]

        self.splits_df_dict["loco_not_in_train_df"] = self.splits_df_dict["no_annotation_df"][self.splits_df_dict["no_annotation_df"][self.seq_col]['0'].isin(loco_not_in_train_df.index)]
        self.splits_df_dict["loco_mixed_df"] = self.splits_df_dict["no_annotation_df"][self.splits_df_dict["no_annotation_df"][self.seq_col]['0'].isin(loco_mixed_df.index)]
        self.splits_df_dict["loco_in_train_df"] = self.splits_df_dict["no_annotation_df"][self.splits_df_dict["no_annotation_df"][self.seq_col]['0'].isin(loco_in_train_df.index)]
        self.splits_df_dict["no_annotation_df"] = self.splits_df_dict["no_annotation_df"][self.splits_df_dict["no_annotation_df"][self.seq_col]['0'].isin(nanas_df.index)]
    
    def get_data(self,path:str,splits:List,ith_run:int = -1):
        #results exist in the outputs folder.
        #outputs folder has two depth levels, first level indicates day and second indicates time per day
        #if path not given, get results from last run
        #ith run specifies the last run (-1), second last(-2)... etc 
        if not path:
            files = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../..', 'outputs'))
            logging.debug(files)
            #newest
            paths = sorted(list(Path(files).rglob('')), key=lambda x: Path.stat(x).st_mtime, reverse=True)
            ith_run = abs(ith_run)
            for path in paths:
                if str(path).endswith('embedds'):
                    ith_run-= 1
                    if ith_run == 0:
                        path = str(path)
                        break

        split_dfs = {}
        splits_to_remove = []
        for split in splits:
            try:
                #read logits csv
                split_df = load(
                    path+f'/{split}_embedds.tsv',
                    header=[0, 1],
                    index_col=0,
                )
                split_df['split','0'] = split
                split_dfs[f"{split}_df"] = split_df

            except:
                splits_to_remove.append(split)
                logger.info(f'{split} does not exist in embedds! Removing it from splits')
        
        for split in splits_to_remove:
            self.splits.remove(split)


        return path,split_dfs

    def get_hp_param(self,hp_param):
        hp_settings = load(path=self.meta_path+'/hp_settings.yaml')
        #hp_param could be in hp_settings .keyes or in a key of a key
        hp_val = hp_settings.get(hp_param)
        if not hp_val:
            for key in hp_settings.keys():
                try:
                    hp_val = hp_settings[key].get(hp_param)
                except:
                    pass
                if hp_val != None:
                    break
        if hp_val == None:
            raise ValueError(f"hp_param {hp_param} not found in hp_settings")

        return hp_val

    def load_mc_mapping_dict(self):
        mapping_dict_path = self.get_hp_param(hp_param="mapping_dict_path")

        return load(mapping_dict_path)

    def compute_umap(self,
        ad,
        nn=50,
        spread=10,
        min_dist=1.0,
    ):
        sc.tl.pca(ad)
        sc.pp.neighbors(ad, n_neighbors=nn, n_pcs=None, use_rep="X_pca")
        sc.tl.umap(ad, n_components=2, spread=spread, min_dist=min_dist)
        logger.info(f'cords are: {ad.obsm}')
        return ad


    def plot_umap(self,ad,
                ncols=3,
                colors=['Labels',"Unseen Labels"],
                edges=False,
                edges_width=0.05,
                run_name = None,
                path=None,
                task=None
        ):
        sc.set_figure_params(dpi = 80,figsize=[10,10])
        fig = sc.pl.umap(
            ad,
            ncols=ncols,
            color=colors,
            edges=edges,
            edges_width=edges_width,
            title=[f"{run_name} approach: {c} {ad.shape}" for c in colors],
            size = ad.obs["size"],
            return_fig=True,
            save=False
        )
        
    #fig.savefig(f'{path}{run_name}_{task}_umap.png')
    def merge_all_splits(self):
        all_dfs = [self.splits_df_dict[f'{split}_df'] for split in self.all_splits]
        self.splits_df_dict['all_df'] = pd.concat(all_dfs).reset_index(drop=True)
        return


def correct_labels(predicted_labels:pd.DataFrame,actual_labels:pd.DataFrame,mapping_dict:Dict):
    '''
    This function corrects the predicted labelsfor the bin based sub classes, tRNAs and miRNAs.
    First both the actual and predicted labels are converted to major class. There are three classes of major classes:
    1. tRNA: if the actual and predicted agree of all the tRNA sub class name except for the last part after -, then the predicted label is corrected to the actual label
    2. bin based sub classes: if the actual and the predicted agree on sub class and the bin number is within 1 of the actual bin number, then the predicted label is corrected to the actual label
    the bin number is after the last -
    3. miRNAs: if the predicted and the actual agree on the first and last part of the subclass, and agree with the numeric part of the middle part, then the predicted label is corrected to the actual label
    '''
    if type(predicted_labels) ==  pd.Series:
        predicted_labels = predicted_labels.values
        actual_labels = actual_labels.values
    import re
    corrected_labels = []
    for i in range(len(predicted_labels)):
        predicted_label = predicted_labels[i]
        actual_label = actual_labels[i]
        if predicted_label == actual_label:
            corrected_labels.append(predicted_label)
        else:
            mc = mapping_dict[actual_label]
            if mc == 'tRNA':
                if mapping_dict[predicted_label] == 'tRNA':
                    predicted_prec = predicted_label.split('__')[1]
                    actual_prec = actual_label.split('__')[1]

                    #the precursor is split by a -, if both have the share the same first part, then correct
                    if predicted_prec == actual_prec:
                        corrected_labels.append(actual_label)
                    else:
                        corrected_labels.append(predicted_label)
                else:
                    corrected_labels.append(predicted_label)
            elif mc == 'miRNA' and ('mir' in predicted_label.lower() or 'let' in predicted_label.lower()):
                #check that the both share the same prime end (either 3p or 5p)
                if predicted_label.split('-')[-1] == actual_label.split('-')[-1]:
                    #check that the both share the same numeric part
                    predicted_numeric = re.findall(r'\d+', predicted_label.split('-')[1])[0]
                    actual_numeric = re.findall(r'\d+', actual_label.split('-')[1])[0]
                    if predicted_numeric == actual_numeric:
                        corrected_labels.append(actual_label)
                    else:
                        corrected_labels.append(predicted_label)
                else:
                    corrected_labels.append(predicted_label)

            elif 'bin' in actual_label:
                if '__' in predicted_label and '__' in actual_label:
                    predicted_label = predicted_label.split('__')[1]
                    actual_label = actual_label.split('__')[1]
                if 'bin' in predicted_label and predicted_label.split('-')[0] == actual_label.split('-')[0]:
                    #get the bin number
                    actual_bin = int(actual_label.split('-')[-1])
                    predicted_bin = int(predicted_label.split('-')[-1])
                    #check that the predicted bin is within 1 of the actual bin
                    if abs(actual_bin - predicted_bin) <= 1:
                        corrected_labels.append(actual_label)
                    else:
                        corrected_labels.append(predicted_label)
                else:
                    corrected_labels.append(predicted_label)
            else:
                corrected_labels.append(predicted_label)
    return corrected_labels
