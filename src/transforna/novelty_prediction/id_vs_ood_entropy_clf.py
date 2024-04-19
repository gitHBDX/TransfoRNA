
#%%
#A script for classifying OOD vs HICO ID (test split). Generates results depicted in figure 4c
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from transforna.utils.file import save
from transforna.utils.tcga_post_analysis_utils import Results_Handler


def entropy_clf(results,random_state:int=1):
    #clf entropy is test vs ood if sub_class else vs loco_not_in_train
    if results.mc_flag:
        ood_ent = results.splits_df_dict["artificial_affix_df"]["Entropy"]["0"].values

    if results.trained_on == 'id':
        test_ent = results.splits_df_dict["test_df"]["Entropy"]["0"].values
    else:
        test_ent = results.splits_df_dict["train_df"]["Entropy"]["0"].values[:int(0.25*len(results.splits_df_dict["train_df"]))]
    ent_x = np.concatenate((ood_ent,test_ent))
    ent_labels = np.concatenate((np.zeros(ood_ent.shape),np.ones(test_ent.shape)))
    trainX, testX, trainy, testy = train_test_split(ent_x, ent_labels, stratify=ent_labels,test_size=0.9, random_state=random_state)

    model = LogisticRegression(solver='lbfgs',class_weight='balanced')

    model.fit(trainX.reshape(-1, 1), trainy)
    #balance testset
    undersample = RandomUnderSampler(sampling_strategy='majority')
    testX,testy = undersample.fit_resample(testX.reshape(-1,1),testy)

    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    yhat = model.predict(testX)
    return testy,lr_probs,yhat,model

def compute_prc(test_labels,lr_probs,yhat,results,show_figure:bool=False):

    lr_precision, lr_recall, _ = precision_recall_curve(test_labels, lr_probs)
    lr_f1, lr_auc = f1_score(test_labels, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    #print(f'{file}: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    if show_figure:
        pyplot.plot(lr_recall, lr_precision, marker='.', label=results.figures_path.split('/')[-2])
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
    # save and show the plot
    plt.title("PRC Curve")

    if results.save_results:
        plt.savefig(f"{results.figures_path}/prc_curve.png")
        plt.savefig(f"{results.figures_path}/prc_curve.svg")

    if show_figure:
        plt.show()
    return lr_f1,lr_auc

def compute_roc(test_labels,lr_probs,results,show_figure:bool=False):
    
    ns_probs = [0 for _ in range(len(test_labels))]

    # calculate scores
    ns_auc = roc_auc_score(test_labels, ns_probs)
    lr_auc = roc_auc_score(test_labels, lr_probs)
    # summarize scores
    #print(f'{file}: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(test_labels, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_labels, lr_probs)

    # plot the roc curve for the model
    if show_figure:
        plt.plot(lr_fpr, lr_tpr, marker='.',markersize=1, label=results.figures_path.split('/')[-2])
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        plt.title("ROC Curve")

    if results.save_results:
        plt.savefig(f"{results.figures_path}/roc_curve.png")
        plt.savefig(f"{results.figures_path}/roc_curve.svg")

    if show_figure:    
        plt.show()
    return lr_auc

def plot_entropy(results):

    entropies_list = []
    test_idx = 0
    for split_idx,split in enumerate(results.splits):
        entropies_list.append(results.splits_df_dict[f"{split}_df"]["Entropy"]["0"].values)
        if split == 'test':
            test_idx = split_idx


    bx = plt.boxplot(entropies_list)
    plt.title("Entropy Distribution")
    plt.ylabel("Entropy")
    plt.xticks(np.arange(len(results.splits))+1,results.splits)
    if results.save_results:
        plt.savefig(f"{results.figures_path}/entropy_test_vs_ood_boxplot.png")
        plt.savefig(f"{results.figures_path}/entropy_test_vs_ood_boxplot.svg")
    plt.xticks(rotation=45)
    plt.show()
    return [item.get_ydata()[1] for item in bx['whiskers']][2*test_idx+1]

    


def plot_entropy_per_unique_length(results,split):
    seqs_len = results.splits_df_dict[f"{split}_df"]["RNA Sequences",'0'].str.len().values
    index = results.splits_df_dict[f"{split}_df"]["RNA Sequences",'0'].values
    entropies = results.splits_df_dict[f"{split}_df"]["Entropy","0"].values

    #create df for plotting
    df = pd.DataFrame({"Entropy":entropies,"Sequences Length":seqs_len},index=index)


    fig = df.boxplot(by='Sequences Length')
    fig.get_figure().gca().set_title("")
    fig.get_figure().gca().set_xlabel(f"Sequences Length ({split})")
    fig.get_figure().gca().set_ylabel("Entropy")
    plt.show() 
    if results.save_results:
        plt.savefig(f"{results.figures_path}/{split}_entropy_per_length_boxplot.png")


def plot_outliers(results,test_whisker_UB):
    test_df = results.splits_df_dict["test_df"]
    test_ent = test_df["Entropy", "0"]
    #decompose outliers in ID
    outlier_seqs = test_df.iloc[(test_whisker_UB < test_ent).values]['RNA Sequences']['0'].values
    outlier_seqs_in_ad = list(set(results.ad.var).intersection(set(outlier_seqs)))
    major_class_dict = results.ad.var.loc[outlier_seqs_in_ad]['small_RNA_class_annotation'][~results.ad.var['hico'].isnull()].value_counts()
    major_class_dict = {x:y for x,y in major_class_dict.items() if y!=0}
    plt.pie(major_class_dict.values(),labels=major_class_dict.keys(),autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    plt.savefig(f"{results.figures_path}/Decomposition_outliers_in_test_pie.png")
    #log outlier seqs to meta
    if results.save_results:
        save(data = outlier_seqs.tolist(),path=results.analysis_path+"/logit_outlier_seqs_ID.yaml")

def log_model_params(model,analysis_path):
    model_params = {"Model Coef": model.coef_[0][0],\
         "Model intercept": model.intercept_[0],\
            "Threshold": -model.intercept_[0]/model.coef_[0][0]}
    model_params = eval(json.dumps(model_params)) 
    save(data = model_params,path=analysis_path+"/logits_model_coef.yaml")

    model.threshold = model_params["Threshold"]
    
def compute_entropy_per_split(results:Results_Handler):
    #compute entropy per split
    for split in results.splits:
        results.splits_df_dict[f"{split}_df"]["Entropy","0"] = entropy(results.splits_df_dict[f"{split}_df"]["Logits"].values,axis=1)
    
def compute_novelty_prediction_per_split(results,model):
    #add noovelty prediction for all splits
    for split in results.splits:
        results.splits_df_dict[f'{split}_df']['Novelty Prediction','is_known_class'] = results.splits_df_dict[f'{split}_df']['Entropy','0']<= model.threshold
    

def compute_logits_clf_metrics(results):
    aucs_roc = []
    aucs_prc = []
    f1s_prc = []
    replicates = 10
    show_figure: bool = False
    for random_state in range(replicates):
        print('random_state: ',random_state)
        #plot only for the last random seed
        if random_state == replicates-1:
            show_figure = True
        #classify ID from OOD using entropy
        test_labels,lr_probs,yhat,model = entropy_clf(results,random_state)
        ###logs
        if results.save_results:
            log_model_params(model,results.analysis_path)
            print("Entropy threshold: ",model.threshold)
            compute_novelty_prediction_per_split(results,model)
            print("Novelty computed")
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

    print(f"auc roc is {auc_roc_score} +- {auc_roc_std}")
    print(f"auc prc is {auc_prc_score} +- {auc_prc_std}")
    print(f"f1 prc is {f1_prc_score} +- {f1_prc_std}")

    logits_clf_metrics = {"AUC ROC score": auc_roc_score,\
        "auc_roc_std": auc_roc_std,\
         "AUC PRC score": auc_prc_score,\
        "auc_prc_std":auc_prc_std,\
            "F1 PRC score": f1_prc_score,\
                "f1_prc_std":f1_prc_std
                }

    logits_clf_metrics = eval(json.dumps(logits_clf_metrics)) 
    if results.save_results:
        save(data = logits_clf_metrics,path=results.analysis_path+"/logits_clf_metrics.yaml")

    


def compute_entropies(trained_on,model):
    #######################################TO CONFIGURE#############################################
    path = ''#f'models/tcga/TransfoRNA_{trained_on.upper()}/sub_class/{model}/embedds' #edit path to contain path for the embedds folder, for example: transforna/results/seq-rev/embedds/
    splits = ['train','valid','test','ood','artificial','no_annotation']
    #run name
    run_name = None #if None, then the name of the model inputs will be used as the name
    #this could be for instance 'Sup Seq-Exp'
    ################################################################################################
    results:Results_Handler = Results_Handler(path=path,splits=splits,read_ad=True,save_results=True)

    results.append_loco_variants()

    results.splits[-1:-1] = ['artificial_affix','loco_not_in_train','loco_mixed','loco_in_train']

    
    if results.mc_flag:
        results.splits.remove("ood")
    
    compute_entropy_per_split(results)
    #remove train and valid from plotting entropy due to clutter
    results.splits.remove("train")
    results.splits.remove("valid")
    
    compute_logits_clf_metrics(results)

    test_whisker_UB = plot_entropy(results)
    print("plotting entropy per unique length")
    plot_entropy_per_unique_length(results,'artificial_affix')
    print('plotting entropy per unique length for ood')
    #decompose outliers in ID
    print("plotting outliers")
    plot_outliers(results,test_whisker_UB)



# %%
