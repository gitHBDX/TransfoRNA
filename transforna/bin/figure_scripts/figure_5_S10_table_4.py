#in this file, the progression of the number of hicos per major class is computed per model
#this is done before ID, after FULL.
#%%
from transforna import load
from transforna import predict_transforna,predict_transforna_all_models
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def compute_overlap_models_ensemble(full_df:pd.DataFrame,mapping_dict:dict):
    full_copy_df = full_df.copy()
    full_copy_df['MC_Labels'] = full_copy_df['Net-Label'].map(mapping_dict)
    #filter is familiar
    full_copy_df = full_copy_df[full_copy_df['Is Familiar?']].set_index('Sequence')
    #count the predicted miRNAs per each Model
    full_copy_df.groupby('Model').MC_Labels.value_counts()

    #for eaach of the models and for each of the mc classes, get the overlap between the models predicting a certain mc and the ensemble having the same prediction
    models = ['Baseline','Seq','Seq-Seq','Seq-Struct','Seq-Rev']
    mcs = full_copy_df.MC_Labels.value_counts().index.tolist()
    mc_stats = {}
    novel_resid = {}
    mcs_predicted_by_only_one_model = {}
    #add all mcs as keys to mc_stats and add all models as keys in every mc
    for mc in mcs:
        mc_stats[mc] = {}
        novel_resid[mc] = {}
        mcs_predicted_by_only_one_model[mc] = {}
        for model in models:
            mc_stats[mc][model] = 0
            novel_resid[mc][model] = 0
            mcs_predicted_by_only_one_model[mc][model] = 0
    
    for mc in mcs:
        ensemble_xrna = full_copy_df[full_copy_df.Model == 'Ensemble'].iloc[full_copy_df[full_copy_df.Model == 'Ensemble'].MC_Labels.str.contains(mc).values].index.tolist()
        for model in models:
            model_xrna = full_copy_df[full_copy_df.Model == model].iloc[full_copy_df[full_copy_df.Model == model].MC_Labels.str.contains(mc).values].index.tolist()
            common_xrna = set(ensemble_xrna).intersection(set(model_xrna))
            try:
                mc_stats[mc][model] = len(common_xrna)/len(ensemble_xrna)
            except ZeroDivisionError:
                mc_stats[mc][model] = 0
            #check how many sequences exist in ensemble but not in model
            try:
                novel_resid[mc][model] = len(set(ensemble_xrna).difference(set(model_xrna)))/len(ensemble_xrna)
            except ZeroDivisionError:
                novel_resid[mc][model] = 0
            #check how many sequences exist in model and in ensemble but not in other models
            other_models_xrna = []
            for other_model in models:
                if other_model != model:
                    other_models_xrna.extend(full_copy_df[full_copy_df.Model == other_model].iloc[full_copy_df[full_copy_df.Model == other_model].MC_Labels.str.contains(mc).values].index.tolist())
            #check how many of model_xrna are not in other_models_xrna and are in ensemble_xrna
            try:    
                mcs_predicted_by_only_one_model[mc][model] = len(set(model_xrna).difference(set(other_models_xrna)).intersection(set(ensemble_xrna)))/len(ensemble_xrna)
            except ZeroDivisionError:
                mcs_predicted_by_only_one_model[mc][model] = 0

    return models,mc_stats,novel_resid,mcs_predicted_by_only_one_model


def plot_bar_overlap_models_ensemble(models,mc_stats,novel_resid,mcs_predicted_by_only_one_model):
    #plot the result as bar plot per mc
    import plotly.graph_objects as go
    import numpy as np
    import plotly.express as px
    #square plot with mc classes on the x axis and the number of hicos on the y axis before ID, after ID, after FULL
    #add cascaded bar plot for novel resid. one per mc per model
    positions = np.arange(len(models))
    fig = go.Figure()
    for model in models:
        fig.add_trace(go.Bar(
            x=list(mc_stats.keys()),
            y=[mc_stats[mc][model] for mc in mc_stats.keys()],
            name=model,
            marker_color=px.colors.qualitative.Plotly[models.index(model)]
                         ))
    
        fig.add_trace(go.Bar(
            x=list(mc_stats.keys()),
            y=[mcs_predicted_by_only_one_model[mc][model] for mc in mc_stats.keys()],
            #base = [mc_stats[mc][model] for mc in mc_stats.keys()],
            name = 'novel',
            marker_color='lightgrey'
        ))
    fig.update_layout(title='Overlap between Ensemble and other models per MC class')

    return fig

def plot_heatmap_overlap_models_ensemble(models,mc_stats,novel_resid,mcs_predicted_by_only_one_model,what_to_plot='overlap'):
    '''
    This function computes a heatmap of the overlap between the ensemble and the other models per mc class
    input:
    models: list of models
    mc_stats: dictionary with mc classes as keys and models as keys of the inner dictionary. values represent overlap between each model and the ensemble
    novel_resid: dictionary with mc classes as keys and models as keys of the inner dictionary. values represent the % of sequences that are predicted by the ensemble as familiar but with specific model as novel
    mcs_predicted_by_only_one_model: dictionary with mc classes as keys and models as keys of the inner dictionary. values represent the % of sequences that are predicted as familiar by only one model
    what_to_plot: string. 'overlap' for overlap between ensemble and other models, 'novel' for novel resid, 'only_one_model' for mcs predicted as novel by only one model

    '''
    
    if what_to_plot == 'overlap':
        plot_dict = mc_stats
    elif what_to_plot == 'novel':
        plot_dict = novel_resid
    elif what_to_plot == 'only_one_model':
        plot_dict = mcs_predicted_by_only_one_model

    import plotly.figure_factory as ff
    fig = ff.create_annotated_heatmap(
        z=[[plot_dict[mc][model] for mc in plot_dict.keys()] for model in models],
        x=list(plot_dict.keys()),
        y=models,
        annotation_text=[[str(round(plot_dict[mc][model],2)) for mc in plot_dict.keys()] for model in models],
        font_colors=['black'],
        colorscale='Blues'
    )
    #set x axis order
    order_x_axis = ['rRNA','tRNA','snoRNA','protein_coding','snRNA','miRNA','lncRNA','piRNA','YRNA','vtRNA']
    fig.update_xaxes(type='category',categoryorder='array',categoryarray=order_x_axis)


    fig.update_xaxes(side='bottom')
    if what_to_plot == 'overlap':
        fig.update_layout(title='Overlap between Ensemble and other models per MC class')
    elif what_to_plot == 'novel':
        fig.update_layout(title='Novel resid between Ensemble and other models per MC class')
    elif what_to_plot == 'only_one_model':
        fig.update_layout(title='MCs predicted by only one model')
    return fig
#%%
#read TCGA
dataset_path_train: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.04.0__var.csv'
models_path = '/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/models/tcga/'
tcga_df = load(dataset_path_train)
tcga_df.set_index('sequence',inplace=True)
loco_hico_na_stats_before = {}
loco_hico_na_stats_before['HICO'] = sum(tcga_df['hico'])/tcga_df.shape[0]
before_hico_seqs = tcga_df['subclass_name'][tcga_df['hico'] == True].index.values
loco_hico_na_stats_before['LOCO'] = (sum(tcga_df.subclass_name != 'no_annotation') - sum(tcga_df['hico']))/tcga_df.shape[0]
before_loco_seqs = tcga_df[tcga_df.hico!=True][tcga_df.subclass_name != 'no_annotation'].index.values
loco_hico_na_stats_before['NA'] = sum(tcga_df.subclass_name == 'no_annotation')/tcga_df.shape[0]
before_na_seqs = tcga_df[tcga_df.subclass_name == 'no_annotation'].index.values
#load mapping dict
mapping_dict_path: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA//data/subclass_to_annotation.json'
mapping_dict = load(mapping_dict_path)
hico_seqs = tcga_df['subclass_name'][tcga_df['hico'] == True].index.values
hicos_mc_before_id_stats = tcga_df.loc[hico_seqs].subclass_name.map(mapping_dict).value_counts()
#remove mcs with ; in them
#hicos_mc_before_id_stats = hicos_mc_before_id_stats[~hicos_mc_before_id_stats.index.str.contains(';')]
seqs_non_hico_id = tcga_df['subclass_name'][tcga_df['hico'] == False].index.values
id_df = predict_transforna(sequences=seqs_non_hico_id,model='Seq-Rev',trained_on='id',path_to_id_models=models_path)
id_df = id_df[id_df['Is Familiar?']].set_index('Sequence')
#print the percentage of sequences with no_annotation and with
print('Percentage of sequences with no annotation: %s'%(id_df[id_df['Net-Label'] == 'no_annotation'].shape[0]/id_df.shape[0]))
print('Percentage of sequences with annotation: %s'%(id_df[id_df['Net-Label'] != 'no_annotation'].shape[0]/id_df.shape[0]))

#%%
hicos_mc_after_id_stats = id_df['Net-Label'].map(mapping_dict).value_counts()
#remove mcs with ; in them
#hicos_mc_after_id_stats = hicos_mc_after_id_stats[~hicos_mc_after_id_stats.index.str.contains(';')]
#add missing major classes with zeros
for mc in hicos_mc_before_id_stats.index:
    if mc not in hicos_mc_after_id_stats.index:
        hicos_mc_after_id_stats[mc] = 0
hicos_mc_after_id_stats = hicos_mc_after_id_stats+hicos_mc_before_id_stats

#%%
seqs_non_hico_full = list(set(seqs_non_hico_id).difference(set(id_df.index.values)))
full_df = predict_transforna_all_models(sequences=seqs_non_hico_full,trained_on='full',path_to_id_models=models_path)
#UNCOMMENT TO COMPUTE BEFORE AND AFTER PER MC: table_4
#ensemble_df = full_df[full_df['Model']=='Ensemble']
#ensemble_df['Major Class'] = ensemble_df['Net-Label'].map(mapping_dict)
#new_hico_mcs= ensemble_df['Major Class'].value_counts()
#ann_hico_mcs = tcga_df[tcga_df['hico'] == True]['small_RNA_class_annotation'].value_counts()

#%%%
inspect_model = True
if inspect_model:
    #from transforna import compute_overlap_models_ensemble,plot_heatmap_overlap_models_ensemble
    models, mc_stats, novel_resid, mcs_predicted_by_only_one_model = compute_overlap_models_ensemble(full_df,mapping_dict)
    fig = plot_heatmap_overlap_models_ensemble(models,mc_stats,novel_resid,mcs_predicted_by_only_one_model,what_to_plot='overlap')
    fig.show()

#%%
df = full_df[full_df.Model == 'Ensemble']
df = df[df['Is Familiar?']].set_index('Sequence')
print('Percentage of sequences with no annotation: %s'%(df[df['Is Familiar?'] == False].shape[0]/df.shape[0]))
print('Percentage of sequences with annotation: %s'%(df[df['Is Familiar?'] == True].shape[0]/df.shape[0]))
hicos_mc_after_full_stats = df['Net-Label'].map(mapping_dict).value_counts()
#remove mcs with ; in them
#hicos_mc_after_full_stats = hicos_mc_after_full_stats[~hicos_mc_after_full_stats.index.str.contains(';')]
#add missing major classes with zeros
for mc in hicos_mc_after_id_stats.index:
    if mc not in hicos_mc_after_full_stats.index:
        hicos_mc_after_full_stats[mc] = 0
hicos_mc_after_full_stats = hicos_mc_after_full_stats + hicos_mc_after_id_stats

# %%
#reorder the index of the series
hicos_mc_before_id_stats = hicos_mc_before_id_stats.reindex(hicos_mc_after_full_stats.index)
hicos_mc_after_id_stats = hicos_mc_after_id_stats.reindex(hicos_mc_after_full_stats.index)
#plot the progression of the number of hicos per major class, before ID, after ID, after FULL as a bar plot
#%%
#%%
training_mcs = ['rRNA','tRNA','snoRNA','protein_coding','snRNA','YRNA','lncRNA']
hicos_mc_before_id_stats_train = hicos_mc_before_id_stats[training_mcs]
hicos_mc_after_id_stats_train = hicos_mc_after_id_stats[training_mcs]
hicos_mc_after_full_stats_train = hicos_mc_after_full_stats[training_mcs]
#plot the progression of the number of hicos per major class, before ID, after ID, after FULL as a bar plot
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import plotly.express as px

#make a square plot with mc classes on the x axis and the number of hicos on the y axis before ID, after ID, after FULL
fig = go.Figure()
fig.add_trace(go.Bar(
    x=hicos_mc_before_id_stats_train.index,
    y=hicos_mc_before_id_stats_train.values,
    name='Before ID',
    marker_color='rgb(31, 119, 180)',
    opacity = 0.5
))
fig.add_trace(go.Bar(
    x=hicos_mc_after_id_stats_train.index,
    y=hicos_mc_after_id_stats_train.values,
    name='After ID',
    marker_color='rgb(31, 119, 180)',
    opacity=0.75
))
fig.add_trace(go.Bar(
    x=hicos_mc_after_full_stats_train.index,
    y=hicos_mc_after_full_stats_train.values,
    name='After FULL',
    marker_color='rgb(31, 119, 180)',
    opacity=1
))
#make log scale
fig.update_layout(
    title='Progression of the Number of HICOs per Major Class',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of HICOs',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title='Major Class',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0.8,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1 
)
#make transparent background
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
#log scalew
fig.update_yaxes(type="log")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
#tilt the x axis labels
fig.update_layout(xaxis_tickangle=22.5)
#set the range of the y axis
fig.update_yaxes(range=[0, 4.5])
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig.write_image("progression_hicos_per_mc_train.svg")
fig.show()
#%%
eval_mcs = ['miRNA','miscRNA','piRNA','vtRNA']
hicos_mc_before_id_stats_eval = hicos_mc_before_id_stats[eval_mcs]
hicos_mc_after_full_stats_eval = hicos_mc_after_full_stats[eval_mcs]

hicos_mc_after_full_stats_eval.index = hicos_mc_after_full_stats_eval.index + '*'
hicos_mc_before_id_stats_eval.index = hicos_mc_before_id_stats_eval.index + '*'
#%%
#plot the progression of the number of hicos per major class, before ID, after ID, after FULL as a bar plot
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import plotly.express as px

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=hicos_mc_before_id_stats_eval.index,
    y=hicos_mc_before_id_stats_eval.values,
    name='Before ID',
    marker_color='rgb(31, 119, 180)',
    opacity = 0.5
))
fig2.add_trace(go.Bar(
    x=hicos_mc_after_full_stats_eval.index,
    y=hicos_mc_after_full_stats_eval.values,
    name='After FULL',
    marker_color='rgb(31, 119, 180)',
    opacity=1
))
#make log scale
fig2.update_layout(
    title='Progression of the Number of HICOs per Major Class',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of HICOs',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title='Major Class',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0.8,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, 
    bargroupgap=0.1 
)
#make transparent background
fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')
#log scalew
fig2.update_yaxes(type="log")

fig2.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
#tilt the x axis labels
fig2.update_layout(xaxis_tickangle=22.5)
#set the range of the y axis
fig2.update_yaxes(range=[0, 4.5])
#adjust bargap
fig2.update_layout(bargap=0.3)
fig2.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
#fig2.write_image("progression_hicos_per_mc_eval.svg")
fig2.show()
# %%
#append df and df_after_id
df_all_hico = df.append(id_df)
loco_hico_na_stats_after = {}
loco_hico_na_stats_after['HICO from NA'] = sum(df_all_hico.index.isin(before_na_seqs))/tcga_df.shape[0]
loco_pred_df = df_all_hico[df_all_hico.index.isin(before_loco_seqs)]
loco_anns_pd = tcga_df.loc[loco_pred_df.index].subclass_name.str.split(';',expand=True)
loco_anns_pd = loco_anns_pd.apply(lambda x: x.str.lower())
#duplicate labels in loco_pred_df * times as the num of columns in loco_anns_pd
loco_pred_labels_df = pd.DataFrame(np.repeat(loco_pred_df['Net-Label'].values,loco_anns_pd.shape[1]).reshape(loco_pred_df.shape[0],loco_anns_pd.shape[1])).set_index(loco_pred_df.index)
loco_pred_labels_df = loco_pred_labels_df.apply(lambda x: x.str.lower())



#%%
trna_mask_df = loco_pred_labels_df.apply(lambda x: x.str.contains('_trna')).any(axis=1)
trna_loco_pred_df = loco_pred_labels_df[trna_mask_df]
#get trna_loco_anns_pd
trna_loco_anns_pd = loco_anns_pd[trna_mask_df]
#for trna_loco_pred_df, remove what prepends the __ and what appends the last -
trna_loco_pred_df = trna_loco_pred_df.apply(lambda x: x.str.split('__').str[1])
trna_loco_pred_df = trna_loco_pred_df.apply(lambda x: x.str.split('-').str[:-1].str.join('-'))
#compute overlap between trna_loco_pred_df and trna_loco_anns_pd
#for every value in trna_loco_pred_df, check if is part of the corresponding position in trna_loco_anns_pd
num_hico_trna_from_loco = 0
for idx,row in trna_loco_pred_df.iterrows():
    trna_label = row[0]
    num_hico_trna_from_loco += trna_loco_anns_pd.loc[idx].apply(lambda x: x!=None and trna_label in x).any()


#%%
#check if 'mir' or 'let' is in any of the values per row. the columns are numbered from 0 to len(loco_anns_pd.columns)
mir_mask_df = loco_pred_labels_df.apply(lambda x: x.str.contains('mir')).any(axis=1) 
let_mask_df = loco_pred_labels_df.apply(lambda x: x.str.contains('let')).any(axis=1)
mir_or_let_mask_df = mir_mask_df | let_mask_df
mir_or_let_loco_pred_df = loco_pred_labels_df[mir_or_let_mask_df]
mir_or_let_loco_anns_pd = loco_anns_pd[mir_or_let_mask_df]
#for each value in mir_or_let_loco_pred_df, if the value contains two '-', remove the last one and what comes after it
mir_or_let_loco_anns_pd = mir_or_let_loco_anns_pd.applymap(lambda x: '-'.join(x.split('-')[:-1]) if x!=None and x.count('-') == 2 else x)
mir_or_let_loco_pred_df = mir_or_let_loco_pred_df.applymap(lambda x: '-'.join(x.split('-')[:-1]) if x!=None and x.count('-') == 2 else x)
#compute overlap between mir_or_let_loco_pred_df and mir_or_let_loco_anns_pd
num_hico_mir_from_loco = sum((mir_or_let_loco_anns_pd == mir_or_let_loco_pred_df).any(axis=1))
#%%


#get rest_loco_anns_pd
rest_loco_pred_df = loco_pred_labels_df[~mir_or_let_mask_df & ~trna_mask_df]
rest_loco_anns_pd = loco_anns_pd[~mir_or_let_mask_df & ~trna_mask_df]

num_hico_bins_from_loco = 0
for idx,row in rest_loco_pred_df.iterrows():
    rest_rna_label = row[0].split('-')[0]
    try:
        bin_no = int(row[0].split('-')[1])
    except:
        continue

    num_hico_bins_from_loco += rest_loco_anns_pd.loc[idx].apply(lambda x: x!=None and rest_rna_label == x.split('-')[0] and abs(int(x.split('-')[1])- bin_no)<=1).any()

loco_hico_na_stats_after['HICO from LOCO'] = (num_hico_trna_from_loco + num_hico_mir_from_loco + num_hico_bins_from_loco)/tcga_df.shape[0]
loco_hico_na_stats_after['LOCO from NA'] = loco_hico_na_stats_before['NA'] - loco_hico_na_stats_after['HICO from NA']
loco_hico_na_stats_after['LOCO from LOCO'] = loco_hico_na_stats_before['LOCO'] - loco_hico_na_stats_after['HICO from LOCO']
loco_hico_na_stats_after['HICO'] = loco_hico_na_stats_before['HICO']

# %%

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

color_mapping = {}
for key in loco_hico_na_stats_before.keys():
    if key.startswith('HICO'):
        color_mapping[key] = "rgb(51,160,44)"
    elif key.startswith('LOCO'):
        color_mapping[key] = "rgb(178,223,138)"
    else:
        color_mapping[key] = "rgb(251,154,153)"
colors = list(color_mapping.values())
fig = go.Figure(data=[go.Pie(labels=list(loco_hico_na_stats_before.keys()), values=list(loco_hico_na_stats_before.values()),hole=.0,marker=dict(colors=colors),sort=False)])
fig.update_layout(title='Percentage of HICOs, LOCOs and NAs before ID')
fig.show()
#save figure as svg
#fig.write_image("pie_chart_before_id.svg")

# %%

color_mapping = {}
for key in loco_hico_na_stats_after.keys():
    if key.startswith('HICO'):
        color_mapping[key] = "rgb(51,160,44)"
    elif key.startswith('LOCO'):
        color_mapping[key] = "rgb(178,223,138)"

loco_hico_na_stats_after = {k: loco_hico_na_stats_after[k] for k in sorted(loco_hico_na_stats_after, key=lambda k: k.startswith('HICO'), reverse=True)}

fig = go.Figure(data=[go.Pie(labels=list(loco_hico_na_stats_after.keys()), values=list(loco_hico_na_stats_after.values()),hole=.0,marker=dict(colors=list(color_mapping.values())),sort=False)])
fig.update_layout(title='Percentage of HICOs, LOCOs and NAs after ID')
fig.show()
#save figure as svg
#fig.write_image("pie_chart_after_id.svg")
