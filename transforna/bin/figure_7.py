   

#%%
#read all files ending with dist_df in bin/lc_files/
import pandas as pd
import glob
from plotly import graph_objects as go
from transforna import load,predict_transforna
all_df = pd.DataFrame()
files = glob.glob('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/transforna/bin/lc_files/*lev_dist_df.csv')
for file in files:
    df = pd.read_csv(file)
    #df['model'] = file.split('/')[-1].split('_')[0]
    all_df = pd.concat([all_df,df])
#drop Unnamed: 0
all_df = all_df.drop(columns=['Unnamed: 0'])
all_df.loc[all_df.split.isnull(),'split'] = 'NA'

#%%
#draw a box plot for the Ensemble model for each of the splits using seaborn
ensemble_df = all_df[all_df.Model == 'Ensemble'].reset_index(drop=True)
#remove other_affixes
ensemble_df = ensemble_df[ensemble_df.split != 'other_affixes'].reset_index(drop=True)
#rename 5'A-affixes to Putative 5’-adapter-prefixes
ensemble_df['split'] = ensemble_df['split'].replace({'5\'A-affixes':'Putative 5’-adapter prefixes','Fused':'Recombined'})
#use plotly to plot boxplots with the following order:
ensemble_df['split'] = ensemble_df['split'].replace({'Relaxed-miRNA':'Isomirs'})
#%%
#plot the boxplot using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize':(15,10)})
sns.set(font_scale=1.5)
order = ['LC-familiar','LC-novel','Random','Putative 5’-adapter prefixes','Recombined','NA','LOCO','Isomirs']
ax = sns.boxplot(x="split", y="NLD", data=ensemble_df, palette="Set3",order=order,showfliers = True)

#add a horizontal line depicting the mean of the Novelty Threshold
ax.axhline(y=ensemble_df['Novelty Threshold'].mean(), color='g', linestyle='--',xmin=0,xmax=1)
#annotate the mean of the Novelty Threshold
ax.annotate('NLD threshold', xy=(1.5, ensemble_df['Novelty Threshold'].mean()), xytext=(1.5, ensemble_df['Novelty Threshold'].mean()-0.07), arrowprops=dict(facecolor='black', shrink=0.05))
#rename Putative 5’-adapter prefixes to 5’-adapter artefacts
ax.set_xticklabels(['LC-Familiar','LC-Novel','Random','5’-adapter artefacts','Recombined','NA','LOCO','IsomiRs'])
#add title
ax.set_facecolor('None')
plt.title('Levenshtein Distance Distribution per Split on LC')
ax.set(xlabel='Split', ylabel='Normalized Levenshtein Distance')
#save legend
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.,facecolor=None,framealpha=0.0)
plt.savefig('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/transforna/bin/lc_figures/lev_dist_no_out_boxplot.svg',dpi=400)
#tilt x axis labels
plt.xticks(rotation=-22.5)
#save svg
plt.savefig('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/transforna/bin/lc_figures/lev_dist_seaboarn_boxplot.svg',dpi=1000)
##save png
plt.savefig('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/transforna/bin/lc_figures/lev_dist_seaboarn_boxplot.png',dpi=1000)
#%%
bars = [r for r in ax.get_children()]
colors = []
for c in bars[:-1]:
    try: colors.append(c.get_facecolor())
    except: pass 
isomir_color = colors[len(order)-1]
isomir_color = [255*x for x in isomir_color]
#covert to rgb('r','g','b','a')
isomir_color = 'rgb(%s,%s,%s,%s)'%(isomir_color[0],isomir_color[1],isomir_color[2],isomir_color[3])

#%%
#filter out the split == 'Relaxed-miRNA'
relaxed_mirna_df = all_df[all_df.split == 'Relaxed-miRNA']
#draw a bar chart of the % of sequences with NLD > Novelty Threshold for each model
#then draw a bar chart using seaborn
models = relaxed_mirna_df.Model.unique()
percentage_dict = {}
for model in models:
    model_df = relaxed_mirna_df[relaxed_mirna_df.Model == model]
    #compute the % of sequences with NLD < Novelty Threshold for each model
    percentage_dict[model] = len(model_df[model_df['NLD'] > model_df['Novelty Threshold']])/len(model_df)
    percentage_dict[model]*=100
#use plotly to plot barplots, all with the same color and with the following order:
#order= ['Baseline','Seq','Seq-Seq','Seq-Struct','Seq-Rev','Ensemble']
fig = go.Figure()
for model in ['Baseline','Seq','Seq-Seq','Seq-Struct','Seq-Rev','Ensemble']:
    fig.add_trace(go.Bar(x=[model],y=[percentage_dict[model]],name=model,marker_color=isomir_color))
    #add percentage on top of each bar
    fig.add_annotation(x=model,y=percentage_dict[model]+2,text='%s%%'%(round(percentage_dict[model],2)),showarrow=False)
    #increase size of annotation
    fig.update_annotations(dict(font_size=13))
#add title in the center
fig.update_layout(title='Percentage of Isomirs considered novel per model')
fig.update_layout(xaxis_tickangle=+22.5)
fig.update_layout(showlegend=False)
#make transparent background
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
#y axis label
fig.update_yaxes(title_text='Percentage of Novel Sequences')
#save svg
fig.show()
#save svg
#fig.write_image('relaxed_mirna_novel_perc_lc_barplot.svg')
#%%
#here we explore the false familiar of the ood lc set
ood_df = pd.read_csv('/nfs/home/yat_ldap/VS_Projects/TransfoRNA/bin/lc_files/LC-novel_lev_dist_df.csv')
mapping_dict_path = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v02/subclass_to_annotation.json'
mapping_dict = load(mapping_dict_path)

LC_path = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v02/LC__ngs__DI_HB_GEL-23.1.2.h5ad'
ad = load(LC_path)
#%%
model = 'Ensemble'
ood_seqs = ood_df[(ood_df.Model == model).values * (ood_df['Is Familiar?'] == True).values].Sequence.tolist()
ood_predicted_labels = ood_df[(ood_df.Model == model).values * (ood_df['Is Familiar?'] == True).values].Labels.tolist()
ood_actual_labels = ad.var.loc[ood_seqs]['subclass_name'].values.tolist()
from transforna import correct_labels
ood_predicted_labels = correct_labels(ood_predicted_labels,ood_actual_labels,mapping_dict)

#get indices where ood_predicted_labels == ood_actual_labels
correct_indices = [i for i, x in enumerate(ood_predicted_labels) if x != ood_actual_labels[i]]
#remove the indices from ood_seqs, ood_predicted_labels, ood_actual_labels
ood_seqs = [ood_seqs[i] for i in correct_indices]
ood_predicted_labels = [ood_predicted_labels[i] for i in correct_indices]
ood_actual_labels = [ood_actual_labels[i] for i in correct_indices]
#get the major class of the actual labels
ood_actual_major_class = [mapping_dict[label] if label in mapping_dict else 'None' for label in ood_actual_labels]
ood_predicted_major_class = [mapping_dict[label] if label in mapping_dict else 'None' for label in ood_predicted_labels ]
#get frequencies of each major class
from collections import Counter
ood_actual_major_class_freq = Counter(ood_actual_major_class)
ood_predicted_major_class_freq = Counter(ood_predicted_major_class)

#plot the length distribution per major class found in ood_actual_major_class


# %%
#use ploty to plot the length distribution of all major classes aggregated
import plotly.express as px
major_classes = list(ood_actual_major_class_freq.keys())

ood_seqs_len = [len(seq) for seq in ood_seqs]
ood_seqs_len_freq = Counter(ood_seqs_len)
fig = px.bar(x=list(ood_seqs_len_freq.keys()),y=list(ood_seqs_len_freq.values()))
fig.show()

#%%
#plot the same length distribution  but cascade the major classes
import plotly.graph_objects as go
fig = go.Figure()
for major_class in major_classes:
    len_dist = [len(ood_seqs[i]) for i, x in enumerate(ood_actual_major_class) if x == major_class]
    len_dist_freq = Counter(len_dist)
    fig.add_trace(go.Bar(x=list(len_dist_freq.keys()),y=list(len_dist_freq.values()),name=major_class))
#stack
fig.update_layout(barmode='stack')
#make transparent background
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
#set y axis label to Count and x axis label to Length
fig.update_layout(yaxis_title='Count',xaxis_title='Length')
#set title
fig.update_layout(title_text="Length distribution of false familiar sequences per major class")
#save as svg
fig.write_image('false_familiar_length_distribution_per_major_class_stacked.svg')
fig.show()

# %%
#for each model, for each split, print Is Familiar? == True and print the number of sequences
for model in all_df.Model.unique():
    print('\n\n')
    model_df = all_df[all_df.Model == model]
    num_hicos = 0
    total_samples = 0
    for split in ['LC-familiar','LC-novel','LOCO','NA','Relaxed-miRNA']:

        split_df = model_df[model_df.split == split]
        #print('Model: %s, Split: %s, Familiar: %s, Number of Sequences: %s'%(model,split,len(split_df[split_df['Is Familiar?'] == True]),len(split_df)))
        #print model, split %
        print('%s %s %s'%(model,split,len(split_df[split_df['Is Familiar?'] == True])/len(split_df)*100))
        if split != 'LC-novel':
            num_hicos+=len(split_df[split_df['Is Familiar?'] == True])
            total_samples+=len(split_df)
    #print % of hicos
    print('%s %s %s'%(model,'HICO',num_hicos/total_samples*100))
    print(total_samples)
# %%
