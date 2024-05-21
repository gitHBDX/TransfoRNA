#in this file, the progression of the number of hicos per major class is computed per model
#this is done before ID, after FULL.
#%%
from transforna import load
from transforna import predict_transforna,predict_transforna_all_models
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import plotly.express as px
mapping_dict_path = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/subclass_to_annotation.json'
models_path = '/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/models/tcga/'

mapping_dict = load(mapping_dict_path)

#%%
dataset:str = 'LC'
hico_loco_na_flag:str = 'hico'
assert hico_loco_na_flag in ['hico','loco_na'], 'hico_loco_na_flag must be either hico or loco_na'
if dataset == 'TCGA':
    dataset_path_train: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.04.0__var.csv'
else:    
    dataset_path_train: str = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v05/2024-04-19__230126_LC_DI_HB_GEL_v23.01.00/sRNA_anno_aggregated_on_seq.csv'

prediction_single_pd = predict_transforna(['AAAAAAACCCCCTTTTTTT'],model='Seq',logits_flag = True,trained_on='id',path_to_id_models=models_path)
sub_classes_used_for_training = prediction_single_pd.columns.tolist()

var = load(dataset_path_train).set_index('sequence')
#remove from var all indexes that are longer than 30
var = var[var.index.str.len() <= 30]
hico_seqs_all = var.index[var['hico']].tolist()
hico_labels_all = var['subclass_name'][var['hico']].values

hico_seqs_id = var.index[var.hico & var.subclass_name.isin(sub_classes_used_for_training)].tolist()
hico_labels_id = var['subclass_name'][var.hico & var.subclass_name.isin(sub_classes_used_for_training)].values

non_hico_seqs = var['subclass_name'][var['hico'] == False].index.values
non_hico_labels = var['subclass_name'][var['hico'] == False].values


#filter hico labels and hico seqs to hico ID
if hico_loco_na_flag == 'loco_na':
    curr_seqs = non_hico_seqs
    curr_labels = non_hico_labels
elif hico_loco_na_flag == 'hico':
    curr_seqs = hico_seqs_id
    curr_labels = hico_labels_id

full_df = predict_transforna_all_models(sequences=curr_seqs,path_to_id_models=models_path)


#%%
mcs = ['rRNA','tRNA','snoRNA','protein_coding','snRNA','miRNA','miscRNA','lncRNA','piRNA','YRNA','vtRNA']
#for each mc, get the sequences of hicos in that mc and compute the number of hicos per model
num_hicos_per_mc = {}
if  hico_loco_na_flag == 'hico':#this is where ground truth exists (hico id)
    curr_labels_id_mc = [mapping_dict[label] for label in curr_labels]

elif hico_loco_na_flag == 'loco_na': # this  is where ground truth does not exist (LOCO/NA)
    ensemble_preds = full_df[full_df.Model == 'Ensemble'].set_index('Sequence').loc[curr_seqs].reset_index()
    curr_labels_id_mc = [mapping_dict[label] for label in ensemble_preds['Net-Label']]

for mc in mcs:
    #select sequences from hico_seqs that are in the major class mc
    mc_seqs = [seq for seq,label in zip(curr_seqs,curr_labels_id_mc) if label == mc]
    if len(mc_seqs) == 0:
        num_hicos_per_mc[mc] = {model:0 for model in full_df.Model.unique()}
        continue
    #only keep in full_df the sequences that are in mc_seqs
    mc_full_df = full_df[full_df.Sequence.isin(mc_seqs)]
    curr_num_hico_per_model = mc_full_df[mc_full_df['Is Familiar?']].groupby(['Model'])['Is Familiar?'].value_counts().droplevel(1)
    #remove Baseline from index
    curr_num_hico_per_model = curr_num_hico_per_model.drop('Baseline')
    curr_num_hico_per_model -= curr_num_hico_per_model.min()
    num_hicos_per_mc[mc] = curr_num_hico_per_model.to_dict()
#%%
to_plot_df = pd.DataFrame(num_hicos_per_mc)
to_plot_mcs = ['rRNA','tRNA','snoRNA']
fig = go.Figure()
#x axis should be the mcs
for  model in num_hicos_per_mc['rRNA'].keys():
    fig.add_trace(go.Bar(x=mcs, y=[num_hicos_per_mc[mc][model] for mc in mcs], name=model))

fig.update_layout(barmode='group')
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
fig.write_image(f"num_hicos_per_model_{dataset}_{hico_loco_na_flag}.svg")
fig.update_yaxes(type="log")
fig.show()

#%%

import pandas as pd
import glob
from plotly import graph_objects as go
from transforna import load,predict_transforna
all_df = pd.DataFrame()
files = glob.glob('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/transforna/bin/lc_files/LC-novel_lev_dist_df.csv')
for file in files:
    df = pd.read_csv(file)
    all_df = pd.concat([all_df,df])
all_df = all_df.drop(columns=['Unnamed: 0'])
all_df.loc[all_df.split.isnull(),'split'] = 'NA'
ensemble_df = all_df[all_df.Model == 'Ensemble']
# %%

lc_path = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v05/2024-04-19__230126_LC_DI_HB_GEL_v23.01.00/sRNA_anno_aggregated_on_seq.csv'
lc_df = load(lc_path)
lc_df.set_index('sequence',inplace=True)
# %%
#filter lc_df to only include sequences that are in ensemble_df
lc_df = lc_df.loc[ensemble_df.Sequence]
actual_major_classes = lc_df['small_RNA_class_annotation']
predicted_major_classes = ensemble_df[['Net-Label','Sequence']].set_index('Sequence').loc[lc_df.index]['Net-Label'].map(mapping_dict)
# %%
#plot correlation matrix between actual and predicted major classes
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
major_classes = list(set(list(predicted_major_classes.unique())+list(actual_major_classes.unique())))
conf_matrix = confusion_matrix(actual_major_classes,predicted_major_classes,labels=major_classes)   
conf_matrix = conf_matrix/np.sum(conf_matrix,axis=1)

sns.heatmap(conf_matrix,annot=True,cmap='Blues')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        conf_matrix[i,j] = round(conf_matrix[i,j],1)


plt.xlabel('Predicted Major Class')
plt.ylabel('Actual Major Class')
plt.xticks(np.arange(len(major_classes)),major_classes,rotation=90)
plt.yticks(np.arange(len(major_classes)),major_classes,rotation=0)
plt.show()

# %%
