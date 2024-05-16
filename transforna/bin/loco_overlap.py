#%%    
import numpy as np
import pandas as pd

import transforna
from transforna import IDModelAugmenter, load

#%%
model_name = 'Baseline'
#read config of model
config_path = f'/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/models/tcga/TransfoRNA_ID/sub_class/{model_name}/meta/hp_settings.yaml'
config = load(config_path)
model_augmenter = IDModelAugmenter(df=None,config=config)
df = model_augmenter.predict_transforna_na()
tcga_df = load('/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.04.0__var.csv')
#set sequence as index
tcga_df.set_index('sequence',inplace=True)
tcga_df['Labels'] = tcga_df['subclass_name'][tcga_df['hico'] == True]
tcga_df['Labels'] = tcga_df['Labels'].astype('category')
#%%
tcga_df.loc[df.Sequence.values,'Labels'] = df['Net-Label'].values

loco_labels_df = tcga_df['subclass_name'].str.split(';',expand=True).loc[df['Sequence']]
#filter the rows having no_annotation in the first row of loco_labels_df
loco_labels_df = loco_labels_df.iloc[~(loco_labels_df[0] == 'no_annotation').values]
#%%
#get the Is Familiar? column from df based on index of loco_labels_df
novelty_prediction_loco_df = df[df['Sequence'].isin(loco_labels_df.index)].set_index('Sequence')['Is Familiar?']
#%%
id_predictions_df = tcga_df.loc[loco_labels_df.index]['Labels']
#copy the columns of id_predictions_df nuber of times equal to the number of columns in loco_labels_df
#name the columns 0 to n
id_predictions_df = pd.concat([id_predictions_df]*loco_labels_df.shape[1],axis=1)
id_predictions_df.columns = np.arange(loco_labels_df.shape[1])
equ_mask = loco_labels_df == id_predictions_df
#check how many rows in eq_mask has atleast one True
num_true = equ_mask.any(axis=1).sum()
print('percentage of all loco RNAs: ',num_true/equ_mask.shape[0])


#now seperate the loco_labels_df into two dataframes based on the coplumn 'Is Familiar?' in df.
#for each value in loco_labels_df, check if it is in classes_used_for_training
fam_loco_labels_df = loco_labels_df[novelty_prediction_loco_df]
#get the rows that are not in train
novel_loco_labels__df = loco_labels_df[~novelty_prediction_loco_df]
#seperate id_predictions_df into two dataframes. novel and familiar
id_predictions_fam_df = id_predictions_df[novelty_prediction_loco_df]
id_predictions_novel_df = id_predictions_df[~novelty_prediction_loco_df]
#%%
num_true_fam = (fam_loco_labels_df == id_predictions_fam_df).any(axis=1).sum()
num_true_novel = (novel_loco_labels__df == id_predictions_novel_df).any(axis=1).sum()
#check how many rows in is_fam_df has atleast one True
print('percentage of similar predictions in familiar: ',num_true_fam/fam_loco_labels_df.shape[0])
print('percentage of similar predictions not in novel: ',num_true_novel/novel_loco_labels__df.shape[0])
print('')
# %%
#filter out fam_loco_labels_df and id_predictions_fam_df based on the rows that are all False in equ_mask
fam_loco_labels_no_overlap_df = fam_loco_labels_df[~equ_mask.any(axis=1)]
id_predictions_fam_no_overlap_df = id_predictions_fam_df[~equ_mask.any(axis=1)]
#collapse the dataframe of fam_loco_labels_df with a ';' seperator
collapsed_loco_labels_df = fam_loco_labels_no_overlap_df.apply(lambda x: ';'.join(x.dropna().astype(str)),axis=1)
#combined collapsed_loco_labels_df with id_predictions_fam_df[0]
predicted_fam_but_ann_novel_df = pd.concat([collapsed_loco_labels_df,id_predictions_fam_no_overlap_df[0]],axis=1)
#rename columns
predicted_fam_but_ann_novel_df.columns = ['KBA_labels','predicted_label']
# %%
#get major class for each column in KBA_labels and predicted_label
mapping_dict_path = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA//data/subclass_to_annotation.json'
sc_to_mc_mapper_dict = load(mapping_dict_path)

predicted_fam_but_ann_novel_df['KBA_labels_mc'] = predicted_fam_but_ann_novel_df['KBA_labels'].str.split(';').apply(lambda x: ';'.join([sc_to_mc_mapper_dict[i] if i in sc_to_mc_mapper_dict.keys() else i for i in x]))
predicted_fam_but_ann_novel_df['predicted_label_mc'] = predicted_fam_but_ann_novel_df['predicted_label'].apply(lambda x: sc_to_mc_mapper_dict[x] if x in sc_to_mc_mapper_dict.keys() else x)
# %%
#for the each of the sequence in predicted_fam_but_ann_novel_df, compute the sim seq along with the lv distance
from transforna.inference_api import predict_transforna

sim_df = predict_transforna(model=model_name,sequences=predicted_fam_but_ann_novel_df.index.tolist(),similarity_flag=True,n_sim=1,trained_on='ID')
sim_df = sim_df.set_index('Sequence')

#append the sim_df to predicted_fam_but_ann_novel_df except for the Labels column
predicted_fam_but_ann_novel_df = pd.concat([predicted_fam_but_ann_novel_df,sim_df.drop('Labels',axis=1)],axis=1)
# %%
#plot the mc proportions of predicted_label_mc
predicted_fam_but_ann_novel_df['predicted_label_mc'].value_counts().plot(kind='bar')
#get order of labels on x axis
x_labels = predicted_fam_but_ann_novel_df['predicted_label_mc'].value_counts().index.tolist()
# %%
#plot the LV distance per predicted_label_mc and order the x axis based on the order of x_labels
fig = predicted_fam_but_ann_novel_df.boxplot(column='NLD',by='predicted_label_mc',figsize=(20,10),rot=90,showfliers=False)
#reorder x axis in fig by x_labels
fig.set_xticklabels(x_labels)
#increase font of axis labels and ticks
fig.set_xlabel('Predicted Label',fontsize=20)
fig.set_ylabel('Levenstein Distance',fontsize=20)
fig.tick_params(axis='both', which='major', labelsize=20)
#display pandas full rows
pd.set_option('display.max_rows', None)