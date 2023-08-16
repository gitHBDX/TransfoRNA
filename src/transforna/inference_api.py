
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

from .utils.file import load
from .utils.tcga_post_analysis_utils import Results_Handler
from .utils.utils import get_closest_ngbr_per_split

from .dataset.dataset_tcga import PrepareGeneData as DatasetTcga
from .utils.utils import (get_hp_setting, get_model, infer_from_pd,
                          prepare_inference_results_tcga)

warnings.filterwarnings("ignore")

def aggregate_ensemble_model(lev_dist_df:pd.DataFrame):
    '''
    This function aggregates the predictions of the ensemble model by choosing the model with the lowest and the highest LV Distance per query sequence.
    If the lowest LV Distance is lower than LV Thresh., then the model with the lowest LV Distance is chosen as the ensemble prediction.
    Otherwise, the model with the highest LV Distance is chosen as the ensemble prediction.
    '''
    #for every sequence, if at least one model scores an LV Distance < LV Thresh., then get the one with the least LV Distance as the ensemble prediction
    #otherwise, get the highest LV Distance.
    #get the minimum LV Distance per query sequence
    #remove the baseline model
    baseline_df = lev_dist_df[lev_dist_df['Model'] == 'Baseline'].reset_index(drop=True)
    lev_dist_df = lev_dist_df[lev_dist_df['Model'] != 'Baseline'].reset_index(drop=True)
    min_lev_dist_df = lev_dist_df.iloc[lev_dist_df.groupby('Sequence')['LV Distance'].idxmin().values]
    #get the maximum LV Distance per query sequence
    max_lev_dist_df = lev_dist_df.iloc[lev_dist_df.groupby('Sequence')['LV Distance'].idxmax().values]
    #choose between each row in min_lev_dist_df and max_lev_dist_df based on the value of LV Thresh.
    novel_mask_df = min_lev_dist_df['LV Distance'] > min_lev_dist_df['LV Thresh.']
    #get the rows where LV Distance is lower than LV Thresh.
    min_lev_dist_df = min_lev_dist_df[~novel_mask_df.values]
    #get the rows where LV Distance is higher than LV Thresh.
    max_lev_dist_df = max_lev_dist_df[novel_mask_df.values]
    #merge min_lev_dist_df and max_lev_dist_df
    ensemble_lev_dist_df = pd.concat([min_lev_dist_df,max_lev_dist_df])
    #add ensemble model
    ensemble_lev_dist_df['Model'] = 'Ensemble'
    #add ensemble_lev_dist_df to lev_dist_df
    lev_dist_df = pd.concat([lev_dist_df,ensemble_lev_dist_df,baseline_df])
    return lev_dist_df.reset_index(drop=True)


def prepare_inference_config(trained_on:str,model:str,mc_or_sc,logits_flag:bool) -> Tuple[Path, DictConfig]:
    root_dir = Path(__file__).parents[1].absolute()

    model_cfg = yaml.load((root_dir / "configs/model/transforna.yaml").read_text(), Loader=SafeLoader)

    infer_cfg = yaml.load((root_dir / "configs/inference_settings/default.yaml").read_text(), Loader=SafeLoader)
    transforna_folder = "TransfoRNA_ID"
    if trained_on == "full":
        transforna_folder = "TransfoRNA_FULL"
    
    if mc_or_sc == 'sc':
        target = 'sub_class'
    else:
        target = 'major_class'
        
    infer_cfg["model_path"] = f"models/tcga/{transforna_folder}/{target}/{model}/ckpt/model_params_tcga.pt"

    print(f'Models used: {transforna_folder}')

    main_cfg = yaml.load((root_dir / "configs/main_config.yaml").read_text(), Loader=SafeLoader)
    main_cfg["model"] = model_cfg
    main_cfg["inference_settings"] = infer_cfg
    main_cfg["model_name"] = model.lower()
    main_cfg["inference"] = True
    main_cfg["log_logits"] = logits_flag

    main_cfg['log_embedds'] = True

    cfg = DictConfig(main_cfg)
    train_cfg_path = get_hp_setting(cfg, "train_config")
    model_cfg_path = get_hp_setting(cfg, "model_config")
    train_config = instantiate(train_cfg_path)
    model_config = instantiate(model_cfg_path)
    # prepare configs as structured dicts
    train_config = OmegaConf.structured(train_config)
    model_config = OmegaConf.structured(model_config)
    # update model config with the name of the model
    model_config["model_input"] = cfg["model_name"]
    return root_dir,OmegaConf.merge({"train_config": train_config, "model_config": model_config}, cfg)
    

def predict_transforna(sequences: List[str], model: str = "Seq-Rev", mc_or_sc:str='sc',\
                    logits_flag:bool = False,attention_flag:bool = False,\
                        similarity_flag:bool=False,n_sim:int=3,embedds_flag:bool = False, \
                            umap_flag:bool = False,trained_on:str='full') -> Tuple:
    
    root_dir,cfg = prepare_inference_config(trained_on,model,mc_or_sc,logits_flag)

    with redirect_stdout(None):
        cfg, net = get_model(cfg, root_dir)
        infer_pd = pd.Series(sequences, name="Sequences").to_frame()
        predicted_labels, logits, gene_embedds_df,attn_scores_pd,all_data, max_len, net = infer_from_pd(cfg, net, infer_pd, DatasetTcga,attention_flag)

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

    else: #return table with predictions, entropy, threshold, is real
        #add aa predictions to infer_pd
        embedds_path = '/'.join(cfg['inference_settings']["model_path"].split('/')[:-2])+'/embedds'
        results:Results_Handler = Results_Handler(path=embedds_path,splits=['train'])
        results.get_knn_model()
        lv_threshold = load(results.analysis_path+"/novelty_model_coef")["Threshold"]
        ent_threshold = load(results.analysis_path+"/logits_model_coef")["Threshold"]
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
            sim_df['Query Sequence'] = sequences_duplicated
            #assign top_5_seqs list to df column
            sim_df[f'Top {n_sim} Similar Sequences'] = top_n_seqs
            sim_df['LV Distance'] = lev_dist
            sim_df['Labels'] = top_n_labels
            sim_df['LV Thresh.'] = lv_threshold
            #for every query sequence, order the LV Distance in a increasing order
            sim_df = sim_df.sort_values(by=['Query Sequence','LV Distance'],ascending=[False,True])
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
            gene_embedds_df['Labels'] = infer_pd['Labels'].values
            gene_embedds_df['Is Familiar?'] = infer_pd['Is Familiar?'].values
            gene_embedds_df['Sequence'] = infer_pd.index
            return gene_embedds_df

        #override threshold
        infer_pd['LV Thresh.'] = lv_threshold
        infer_pd['LV Distance'] = lv_dist_closest
        infer_pd['Ent Thresh.'] = ent_threshold
        infer_pd = infer_pd.round({"LV Distance": 2, "LV Thresh.": 2})
        print(f'num of new hico based on levenstein distance is {np.sum(infer_pd["Is Familiar?"])}')
        return infer_pd.rename_axis("Sequence").reset_index()

def predict_transforna_all_models(sequences: List[str], mc_or_sc:str = 'sc',logits_flag: bool = False, attention_flag: bool = False,\
        similarity_flag: bool = False, n_sim:int = 3,
        embedds_flag:bool=False, umap_flag:bool = False, trained_on:str="full"):
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
        df_ = predict_transforna(sequences, model, mc_or_sc,logits_flag,attention_flag,similarity_flag,n_sim,embedds_flag,umap_flag,trained_on=trained_on)
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
