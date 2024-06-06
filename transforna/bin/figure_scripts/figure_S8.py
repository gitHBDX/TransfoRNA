#%%
from transforna import load
from transforna import Results_Handler
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
path = '/media/ftp_share/hbdx/analysis/tcga/TransfoRNA_I_ID_V4/sub_class/Seq/embedds/'
results:Results_Handler = Results_Handler(path=path,splits=['train','valid','test','ood'],read_ad=True)

mapping_dict = load('/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v02/subclass_to_annotation.json')
mapping_dict['artificial_affix'] = 'artificial_affix'
train_df = results.splits_df_dict['train_df']
valid_df = results.splits_df_dict['valid_df']
test_df = results.splits_df_dict['test_df']
ood_df = results.splits_df_dict['ood_df']
#remove RNA Sequences from the dataframe if not in results.ad.var.index
train_df = train_df[train_df['RNA Sequences'].isin(results.ad.var[results.ad.var['hico'] == True].index)['0'].values]
valid_df = valid_df[valid_df['RNA Sequences'].isin(results.ad.var.index[results.ad.var['hico'] == True])['0'].values]
test_df = test_df[test_df['RNA Sequences'].isin(results.ad.var.index[results.ad.var['hico'] == True])['0'].values]
ood_df = ood_df[ood_df['RNA Sequences'].isin(results.ad.var.index[results.ad.var['hico'] == True])['0'].values]
#concatenate train,valid and test
train_val_test_df = train_df.append(valid_df).append(test_df)
#map Labels to annotation
hico_id_labels = train_val_test_df['Labels','0'].map(mapping_dict).values
hico_ood_labels = ood_df['Labels','0'].map(mapping_dict).values
#read ad
ad = results.ad

hico_loco_df = pd.DataFrame(columns=['mc','hico_id','hico_ood','loco'])
for mc in ad.var['small_RNA_class_annotation'][ad.var['hico'] == True].unique():
    hico_loco_df = hico_loco_df.append({'mc':mc,
                    'hico_id':sum([mc in i for i in hico_id_labels]),
                    'hico_ood':sum([mc in i for i in hico_ood_labels]),
                    'loco':sum([mc in i for i in ad.var['small_RNA_class_annotation'][ad.var['hico'] != True].values])},ignore_index=True)  
# %%


order = ['rRNA','tRNA','snoRNA','protein_coding','snRNA','miRNA','miscRNA','lncRNA','piRNA','YRNA','vtRNA']

fig = go.Figure()
fig.add_trace(go.Bar(x=hico_loco_df['mc'],y=hico_loco_df['hico_id'],name='HICO ID',marker_color='#00CC96'))
fig.add_trace(go.Bar(x=hico_loco_df['mc'],y=hico_loco_df['hico_ood'],name='HICO OOD',marker_color='darkcyan'))
fig.add_trace(go.Bar(x=hico_loco_df['mc'],y=hico_loco_df['loco'],name='LOCO',marker_color='#7f7f7f'))
fig.update_layout(barmode='group')
fig.update_layout(width=800,height=800)
#order the x axis
fig.update_layout(xaxis={'categoryorder':'array','categoryarray':order})
fig.update_layout(xaxis_title='Major Class',yaxis_title='Number of Sequences')
fig.update_layout(title='Number of Sequences per Major Class in ID, OOD and LOCO')
fig.update_layout(yaxis_type="log")
#save as png
pio.write_image(fig,'hico_id_ood_loco_proportion.png')
#save as svg
pio.write_image(fig,'hico_id_ood_loco_proportion.svg')
fig.show()
# %%
