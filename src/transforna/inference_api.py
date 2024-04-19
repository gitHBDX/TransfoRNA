
import warnings
from argparse import ArgumentParser
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from yaml.loader import SafeLoader

from .processing.seq_tokenizer import SeqTokenizer
from .utils.file import load
from .utils.tcga_post_analysis_utils import Results_Handler
from .utils.utils import (get_model, infer_from_pd,
                          prepare_inference_results_tcga,
                          update_config_with_inference_params)

from .novelty_prediction.id_vs_ood_nld_clf import get_closest_ngbr_per_split

warnings.filterwarnings("ignore")

def aggregate_ensemble_model(lev_dist_df:pd.DataFrame):
    '''
    This function aggregates the predictions of the ensemble model by choosing the model with the lowest and the highest NLD per query sequence.
    If the lowest NLD is lower than Novelty Threshold, then the model with the lowest NLD is chosen as the ensemble prediction.
    Otherwise, the model with the highest NLD is chosen as the ensemble prediction.
    '''
    #for every sequence, if at least one model scores an NLD < Novelty Threshold, then get the one with the least NLD as the ensemble prediction
    #otherwise, get the highest NLD.
    #get the minimum NLD per query sequence
    #remove the baseline model
    baseline_df = lev_dist_df[lev_dist_df['Model'] == 'Baseline'].reset_index(drop=True)
    lev_dist_df = lev_dist_df[lev_dist_df['Model'] != 'Baseline'].reset_index(drop=True)
    min_lev_dist_df = lev_dist_df.iloc[lev_dist_df.groupby('Sequence')['NLD'].idxmin().values]
    #get the maximum NLD per query sequence
    max_lev_dist_df = lev_dist_df.iloc[lev_dist_df.groupby('Sequence')['NLD'].idxmax().values]
    #choose between each row in min_lev_dist_df and max_lev_dist_df based on the value of Novelty Threshold
    novel_mask_df = min_lev_dist_df['NLD'] > min_lev_dist_df['Novelty Threshold']
    #get the rows where NLD is lower than Novelty Threshold
    min_lev_dist_df = min_lev_dist_df[~novel_mask_df.values]
    #get the rows where NLD is higher than Novelty Threshold
    max_lev_dist_df = max_lev_dist_df[novel_mask_df.values]
    #merge min_lev_dist_df and max_lev_dist_df
    ensemble_lev_dist_df = pd.concat([min_lev_dist_df,max_lev_dist_df])
    #add ensemble model
    ensemble_lev_dist_df['Model'] = 'Ensemble'
    #add ensemble_lev_dist_df to lev_dist_df
    lev_dist_df = pd.concat([lev_dist_df,ensemble_lev_dist_df,baseline_df])
    return lev_dist_df.reset_index(drop=True)


def read_inference_model_config(model:str,mc_or_sc,trained_on:str,path_to_models:str):
    transforna_folder = "TransfoRNA_ID"
    if trained_on == "full":
        transforna_folder = "TransfoRNA_FULL"
    if mc_or_sc == 'sc':
        target = 'sub_class'
    else:
        target = 'major_class'

    model_path = f"{path_to_models}/{transforna_folder}/{target}/{model}/meta/hp_settings.yaml"
    cfg = OmegaConf.load(model_path)
    return cfg

def predict_transforna(sequences: List[str], model: str = "Seq-Rev", mc_or_sc:str='sc',\
                    logits_flag:bool = False,attention_flag:bool = False,\
                        similarity_flag:bool=False,n_sim:int=3,embedds_flag:bool = False, \
                            umap_flag:bool = False,trained_on:str='full',path_to_models:str='') -> pd.DataFrame:
    '''
    This function predicts the major class or sub class of a list of sequences using the TransfoRNA model.
    Additionaly, it can return logits, attention scores, similarity scores, gene embeddings or umap embeddings.

    Input:
        sequences: list of sequences to predict
        model: model to use for prediction
        mc_or_sc: models trained on major class or sub class
        logits_flag: whether to return logits
        attention_flag: whether to return attention scores (obtained from the self-attention layer)
        similarity_flag: whether to return explanatory/similar sequences in the training set
        n_sim: number of similar sequences to return
        embedds_flag: whether to return embeddings of the sequences
        umap_flag: whether to return umap embeddings
        trained_on: whether to use the model trained on the full dataset or the ID dataset
    Output:
        pd.DataFrame with the predictions
    '''
    #assers that only one flag is True
    assert sum([logits_flag,attention_flag,similarity_flag,embedds_flag,umap_flag]) <= 1, 'One option at most can be True'
    # capitalize the first letter of the model and the first letter after the -
    model = "-".join([word.capitalize() for word in model.split("-")])
    cfg = read_inference_model_config(model,mc_or_sc,trained_on,path_to_models)
    cfg = update_config_with_inference_params(cfg,mc_or_sc,trained_on,path_to_models)
    root_dir = Path(__file__).parents[1].absolute()

    with redirect_stdout(None):
        cfg, net = get_model(cfg, root_dir)
        infer_pd = pd.Series(sequences, name="Sequences").to_frame()
        predicted_labels, logits, gene_embedds_df,attn_scores_pd,all_data, max_len, net = infer_from_pd(cfg, net, infer_pd, SeqTokenizer,attention_flag)

        if model == 'Seq':
            gene_embedds_df = gene_embedds_df.iloc[:,:int(gene_embedds_df.shape[1]/2)]

    prepare_inference_results_tcga(cfg, predicted_labels, logits, all_data, max_len)
    infer_pd = all_data["infere_rna_seq"]

    if logits_flag:
        logits_df = infer_pd.rename_axis("Sequence").reset_index()
        logits_cols = [col for col in logits_df.columns if "Logits" in col]
        logits_df = logits_df[logits_cols]
        logits_df.columns = pd.MultiIndex.from_tuples(logits_df.columns, names=["Logits", "Sub Class"])
        logits_df.columns = logits_df.columns.droplevel(0)
        return logits_df
    
    elif attention_flag:
        return attn_scores_pd

    elif embedds_flag:
        return gene_embedds_df

    else: #return table with predictions, entropy, threshold, is familiar
        #add aa predictions to infer_pd
        embedds_path = '/'.join(cfg['inference_settings']["model_path"].split('/')[:-2])+'/embedds'
        results:Results_Handler = Results_Handler(path=embedds_path,splits=['train'])
        results.get_knn_model()
        lv_threshold = load(results.analysis_path+"/novelty_model_coef")["Threshold"]
        print(f'computing levenstein distance for the inference set')
        #prepare infer split
        gene_embedds_df.columns = results.embedds_cols[:len(gene_embedds_df.columns)]
        #add index of gene_embedds_df to be a column with name results.seq_col
        gene_embedds_df[results.seq_col] = gene_embedds_df.index
        #set gene_embedds_df as the new infer split
        results.splits_df_dict['infer_df'] = gene_embedds_df


        _,_,top_n_seqs,top_n_labels,distances,lev_dist = get_closest_ngbr_per_split(results,'infer',num_neighbors=n_sim)

        if similarity_flag:
            #create df
            sim_df = pd.DataFrame()
            #populate query sequences and duplicate them n times
            sequences = gene_embedds_df.index.tolist()
            #duplicate each sequence n_sim times
            sequences_duplicated = [seq for seq in sequences for _ in range(n_sim)]
            sim_df['Sequence'] = sequences_duplicated
            #assign top_5_seqs list to df column
            sim_df[f'Explanatory Sequence'] = top_n_seqs
            sim_df['NLD'] = lev_dist
            sim_df['Labels'] = top_n_labels
            sim_df['Novelty Threshold'] = lv_threshold
            #for every query sequence, order the NLD in a increasing order
            sim_df = sim_df.sort_values(by=['Sequence','NLD'],ascending=[False,True])
            return sim_df
        
        print(f'num of hico based on entropy novelty prediction is {sum(infer_pd["Is Familiar?"])}')
        #for every n_sim elements in the list, get the smallest levenstein distance 
        lv_dist_closest = [min(lev_dist[i:i+n_sim]) for i in range(0,len(lev_dist),n_sim)]
        infer_pd['Is Familiar?'] = [True if lv<lv_threshold else False for lv in lv_dist_closest]

        if umap_flag:
            #compute umap
            print(f'computing umap for the inference set')
            gene_embedds_df = gene_embedds_df.drop(results.seq_col,axis=1)
            umap = UMAP(n_components=2,random_state=42)
            scaled_embedds = StandardScaler().fit_transform(gene_embedds_df.values)
            gene_embedds_df = pd.DataFrame(umap.fit_transform(scaled_embedds),columns=['UMAP1','UMAP2'])
            gene_embedds_df['Net-Label'] = infer_pd['Net-Label'].values
            gene_embedds_df['Is Familiar?'] = infer_pd['Is Familiar?'].values
            gene_embedds_df['Sequence'] = infer_pd.index
            return gene_embedds_df

        #override threshold
        infer_pd['Novelty Threshold'] = lv_threshold
        infer_pd['NLD'] = lv_dist_closest
        infer_pd = infer_pd.round({"NLD": 2, "Novelty Threshold": 2})
        print(f'num of new hico based on levenstein distance is {np.sum(infer_pd["Is Familiar?"])}')
        return infer_pd.rename_axis("Sequence").reset_index()

def predict_transforna_all_models(sequences: List[str], mc_or_sc:str = 'sc',logits_flag: bool = False, attention_flag: bool = False,\
        similarity_flag: bool = False, n_sim:int = 3,
        embedds_flag:bool=False, umap_flag:bool = False, trained_on:str="full",path_to_models:str='') -> pd.DataFrame:
    """
    Predicts the labels of the sequences using all the models available in the transforna package.
    If non of the flags are true, it constructs and aggrgates the output of the ensemble model.
    
    Input:
        sequences: list of sequences to predict
        mc_or_sc: models trained on major class or sub class
        logits_flag: whether to return logits
        attention_flag: whether to return attention scores (obtained from the self-attention layer)
        similarity_flag: whether to return explanatory/similar sequences in the training set
        n_sim: number of similar sequences to return
        embedds_flag: whether to return embeddings of the sequences
        umap_flag: whether to return umap embeddings
        trained_on: whether to use the model trained on the full dataset or the ID dataset
    Output:
        df: dataframe with the predictions
    """
    #print time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Before", current_time)
    models = ["Baseline","Seq", "Seq-Seq", "Seq-Struct", "Seq-Rev"]
    if similarity_flag or embedds_flag: #remove baseline, takes long time
        models = ["Baseline","Seq", "Seq-Seq", "Seq-Struct", "Seq-Rev"]
    if attention_flag: #remove single based transformer models
        models = ["Seq", "Seq-Struct", "Seq-Rev"]
    df = None
    for model in models:
        print(model)
        df_ = predict_transforna(sequences, model, mc_or_sc,logits_flag,attention_flag,similarity_flag,n_sim,embedds_flag,umap_flag,trained_on=trained_on,path_to_models = path_to_models)
        df_["Model"] = model
        df = pd.concat([df, df_], axis=0)
    #aggregate ensemble model if not of the flags are true
    if not logits_flag and not attention_flag and not similarity_flag and not embedds_flag and not umap_flag:
        df = aggregate_ensemble_model(df)
    #print time after inference
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("After", current_time)
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--logits_flag", nargs="?", const = True,default=False)
    parser.add_argument("--attention_flag", nargs="?", const = True,default=False)
    parser.add_argument("--similarity_flag", nargs="?", const = True,default=False)
    parser.add_argument("--n_sim", nargs="?", const = 3,default=3)
    parser.add_argument("--embedds_flag", nargs="?", const = True,default=False)
    parser.add_argument("--trained_on", nargs="?", const = True,default="full")
    predict_transforna_all_models(**vars(parser.parse_args()))
