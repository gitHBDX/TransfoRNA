
# %%
#A script for classifying OOD vs HICO ID (test split). Generates results depicted in figure 4c

import json
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from imblearn.under_sampling import RandomUnderSampler
from Levenshtein import distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..utils.file import load, save
from ..utils.tcga_post_analysis_utils import Results_Handler
from .utlis import compute_prc, compute_roc

logger = logging.getLogger(__name__)

def get_lev_dist(seqs_a_list:List[str],seqs_b_list:List[str]):
    '''
    compute levenstein distance between two lists of sequences and normalize by the length of the longest sequence
    The lev distance is computed between seqs_a_list[i] and seqs_b_list[i]
    '''
    lev_dist = []
    for i in range(len(seqs_a_list)):
        dist = distance(seqs_a_list[i],seqs_b_list[i])
        #normalize
        dist = dist/max(len(seqs_a_list[i]),len(seqs_b_list[i]))
        lev_dist.append(dist)
    return lev_dist

def get_closest_neighbors(results:Results_Handler,query_embedds:np.ndarray,num_neighbors:int=1):
    '''
    get the closest neighbors to the query embedds using the knn model in results
    The closest neighbors are to be found in the training set
    '''
    #norm infer embedds
    query_embedds = query_embedds/np.linalg.norm(query_embedds,axis=1)[:,None]
    #get top 1 seqs
    distances, indices = results.knn_model.kneighbors(query_embedds)
    distances = distances[:,:num_neighbors].flatten()
    #flatten distances

    indices = indices[:,:num_neighbors]

    top_n_seqs = np.array(results.knn_seqs)[indices][:,:num_neighbors]
    top_n_seqs = [seq[0] for sublist in top_n_seqs for seq in sublist]
    top_n_labels = np.array(results.knn_labels)[indices][:,:num_neighbors]
    top_n_labels = [label[0] for sublist in top_n_labels for label in sublist]
    
    return top_n_seqs,top_n_labels,distances

def get_closest_ngbr_per_split(results:Results_Handler,split:str,num_neighbors:int=1):
    '''
    compute levenstein distance between the sequences in split and their closest neighbors in the training set
    '''
    split_df = results.splits_df_dict[f'{split}_df']
    #log
    logger.debug(f'number of sequences in {split} is {split_df.shape[0]}')
    #accomodate for multi-index df or single index
    try:
        split_seqs = split_df[results.seq_col].values[:,0]
    except:
        split_seqs = split_df[results.seq_col].values
    try:
        split_labels = split_df[results.label_col].values[:,0]
    except:
        split_labels = None
    #get embedds
    embedds = split_df[results.embedds_cols].values
    
    top_n_seqs,top_n_labels,distances = get_closest_neighbors(results,embedds,num_neighbors)
    #get levenstein distance
    #for each split_seqs duplicate it num_neighbors times
    split_seqs = [seq for seq in split_seqs for _ in range(num_neighbors)]
    lev_dist = get_lev_dist(split_seqs,top_n_seqs)
    return split_seqs,split_labels,top_n_seqs,top_n_labels,distances,lev_dist


def log_lev_params(threshold:float,analysis_path:str):
    model_params = {"Threshold": threshold}
    model_params = eval(json.dumps(model_params)) 
    save(data = model_params,path=analysis_path+"/novelty_model_coef.yaml")

def lev_clf(set_a,set_b,random_state):
    #get labels
    y = np.concatenate((np.zeros(len(set_a)),np.ones(len(set_b))))
    #get levenstein distance
    lev_dist = np.concatenate((set_a,set_b))
    #upsample minority class
    oversample = RandomUnderSampler(sampling_strategy='majority',random_state=random_state)
    lev_dist, y = oversample.fit_resample(lev_dist.reshape(-1,1), y)
    #get levenstein distance as a feature
    lev_dist = lev_dist.reshape(-1,1)
    #split to train and test
    X_train, X_test, y_train, y_test = train_test_split(lev_dist, y, test_size=0.33, random_state=random_state)
    #define model
    model = LogisticRegression(solver='lbfgs')
    #fit model
    model.fit(X_train, y_train)
    #predict probabilities
    lr_probs = model.predict_proba(X_test)[:, 1]
    #predict class
    yhat = model.predict(X_test)
    return y_test,lr_probs,yhat,model


def compute_novelty_clf_metrics(results:Results_Handler,lev_dist_id_set,lev_dist_ood_set):
    aucs_roc = []
    aucs_prc = []
    f1s_prc = []
    thresholds = []
    replicates = 10
    show_figure: bool = False

    for random_state in range(replicates):
        #plot only for the last random seed
        if random_state == replicates-1:
            show_figure = True
        #classify ID from OOD using entropy
        test_labels,lr_probs,yhat,model = lev_clf(lev_dist_id_set,lev_dist_ood_set,random_state)
        thresholds.append(-model.intercept_[0]/model.coef_[0][0])
        mean_thresh = sum(thresholds)/len(thresholds)
        ###logs
        if results.save_results:
            log_lev_params(mean_thresh,results.analysis_path)
        ###plots 
        auc_roc = compute_roc(test_labels,lr_probs,results,show_figure)
        f1_prc,auc_prc = compute_prc(test_labels,lr_probs,yhat,results,show_figure)
        aucs_roc.append(auc_roc)
        aucs_prc.append(auc_prc)
        f1s_prc.append(f1_prc)


    auc_roc_score = sum(aucs_roc)/len(aucs_roc)
    auc_roc_std = np.std(aucs_roc)
    auc_prc_score = sum(aucs_prc)/len(aucs_prc)
    auc_prc_std = np.std(aucs_prc)
    f1_prc_score = sum(f1s_prc)/len(f1s_prc)
    f1_prc_std = np.std(f1s_prc)

    logger.info(f"auc roc is {auc_roc_score} +- {auc_roc_std}")
    logger.info(f"auc prc is {auc_prc_score} +- {auc_prc_std}")
    logger.info(f"f1 prc is {f1_prc_score} +- {f1_prc_std}")

    novelty_clf_metrics = {"AUC ROC score": auc_roc_score,\
        "auc_roc_std": auc_roc_std,\
         "AUC PRC score": auc_prc_score,\
        "auc_prc_std":auc_prc_std,\
            "F1 PRC score": f1_prc_score,\
                "f1_prc_std":f1_prc_std
                }

    novelty_clf_metrics = eval(json.dumps(novelty_clf_metrics)) 
    if results.save_results:
        save(data = novelty_clf_metrics,path=results.analysis_path+"/novelty_clf_metrics.yaml")
    
    return sum(thresholds)/len(thresholds)


def compute_nlds(embedds_path):
    logger.info("Computing NLD metrics")
    #######################################TO CONFIGURE#############################################
    logger.info("Computing novelty clf metrics")
    #embedds_path = ''#f'models/tcga/TransfoRNA_{trained_on.upper()}/sub_class/{model}/embedds' #edit path to contain path for the embedds folder, for example: transforna/results/seq-rev/embedds/
    splits = ['train','valid','test','ood','artificial','no_annotation']
    #run name
    run_name = None #if None, then the name of the model inputs will be used as the name
    #this could be for instance 'Sup Seq-Exp'
    ################################################################################################
    results:Results_Handler = Results_Handler(embedds_path=embedds_path,splits=splits,read_dataset=True,create_knn_graph=True,save_results=True)
    results.append_loco_variants()
    #get knn model
    results.get_knn_model()
    lev_dist_df = pd.DataFrame()
    
    
    #compute levenstein distance per split
    for split in results.splits_df_dict.keys():
        if len(results.splits_df_dict[f'{split}']) == 0:
            continue
        split_seqs,split_labels,top_n_seqs,top_n_labels,distances,lev_dist = get_closest_ngbr_per_split(results,'_'.join(split.split('_')[:-1]))
        #create df from split and levenstein distance
        lev_dist_split_df = pd.DataFrame({'split':split,'lev_dist':lev_dist,'seqs':split_seqs,'labels':split_labels,'top_n_seqs':top_n_seqs,'top_n_labels':top_n_labels})
        #append to lev_dist_df
        lev_dist_df = lev_dist_df.append(lev_dist_split_df)

    #plot boxplot levenstein distance per split using plotly and add seqs and labels
    fig = px.box(lev_dist_df, x="split", y="lev_dist",points="all",hover_data=['seqs','labels','top_n_seqs','top_n_labels'])
    #reduce marker size
    fig.update_traces(marker=dict(size=2))
    fig.show()
    #save as html file in figures_path
    fig.write_html(f'{results.figures_path}/lev_distance_distribution.html')
    fig.write_image(f'{results.figures_path}/lev_distance_distribution.png')

    #get rows of lev_dist_df from ood/artificial_affix split and from test split
    if 'ood_df' in lev_dist_df['split'].values: #for ID models
        novel_df = lev_dist_df[lev_dist_df['split'] == 'ood_df']
    else:#for FULL models as all classes are used for training: no OOD
        novel_df = lev_dist_df[lev_dist_df['split'] == 'artificial_affix_df']
    test_df = lev_dist_df[lev_dist_df['split'] == 'test_df']
    
    lev_dist_df.to_csv(f'{results.analysis_path}/lev_dist_df.csv')
   
    #compute novelty clf metrics
    compute_novelty_clf_metrics(results,test_df['lev_dist'].values,novel_df['lev_dist'].values)






    



