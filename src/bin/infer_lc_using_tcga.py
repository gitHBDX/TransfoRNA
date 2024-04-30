import random
import sys
from random import randint

import pandas as pd
import plotly.graph_objects as go

from transforna.inference_api import (predict_transforna,
                                      predict_transforna_all_models)
from transforna.utils.file import load
from transforna.utils.tcga_post_analysis_utils import (Results_Handler,
                                                       correct_labels)
from transforna.utils.utils import get_fused_seqs


def get_mc_sc(ad,sequences,sub_classes_used_for_training,sc_to_mc_mapper_dict,ood_flag = False):

    infered_seqs_var = ad.var.loc[sequences]
    sc_classes_df = infered_seqs_var['subclass_name'].str.split(';',expand=True)
    #filter rows with all nans in sc_classes_df
    sc_classes_df = sc_classes_df[~sc_classes_df.isnull().all(axis=1)]
    #cmask for classes used for training
    if ood_flag:
        mask = sc_classes_df.applymap(lambda x: True if x not in sub_classes_used_for_training or pd.isnull(x) else False)
    else:
        mask = sc_classes_df.applymap(lambda x: True if x in sub_classes_used_for_training or pd.isnull(x) else False)
    
    #check if any sub class in sub_classes_used_for_training is in sc_classes_df
    if mask.apply(lambda x: all(x.tolist()), axis=1).sum() == 0:
        #TODO: change to log
        import logging
        log_ = logging.getLogger(__name__)
        log_.error('None of the sub classes used for training are in the sequences')
        raise Exception('None of the sub classes used for training are in the sequences')

    #filter rows with atleast one False in mask
    sc_classes_df = sc_classes_df[mask.apply(lambda x: all(x.tolist()), axis=1)]
    #get mc classes
    mc_classes_df = sc_classes_df.applymap(lambda x: sc_to_mc_mapper_dict[x] if x in sc_to_mc_mapper_dict else x)
    #assign major class for not found values if containing 'miRNA', 'tRNA', 'rRNA', 'snRNA', 'snoRNA'
    #mc_classes_df = mc_classes_df.applymap(lambda x: None if x is None else 'miRNA' if 'miR' in x else 'tRNA' if 'tRNA' in x else 'rRNA' if 'rRNA' in x else 'snRNA' if 'snRNA' in x else 'snoRNA' if 'snoRNA' in x else 'snoRNA' if 'SNO' in x else 'protein_coding' if 'RPL37A' in x else 'lncRNA' if 'SNHG1' in x else 'not_found')
    #filter all 'not_found' rows
    mc_classes_df = mc_classes_df[mc_classes_df.apply(lambda x: 'not_found' not in x.tolist(), axis=1)]
    #filter index
    sc_classes_df = sc_classes_df.loc[mc_classes_df.index]
    mc_classes_df = mc_classes_df.loc[sc_classes_df.index]
    return mc_classes_df,sc_classes_df
    
def plot_confusion_false_novel(df,sc_df,mc_df,save_figs:bool=False):
    #filter index of sc_classes_df to contain indices of outliers df
    curr_sc_classes_df = sc_df.loc[[i for i in df.index if i in sc_df.index]]
    curr_mc_classes_df = mc_df.loc[[i for i in df.index if i in mc_df.index]]
    #convert Labels to mc_Labels
    df = df.assign(predicted_mc_labels=df.apply(lambda x: sc_to_mc_mapper_dict[x['predicted_sc_labels']] if x['predicted_sc_labels'] in sc_to_mc_mapper_dict else 'miRNA' if 'miR' in x['predicted_sc_labels'] else 'tRNA' if 'tRNA' in x['predicted_sc_labels'] else 'rRNA' if 'rRNA' in x['predicted_sc_labels'] else 'snRNA' if 'snRNA' in x['predicted_sc_labels'] else 'snoRNA' if 'snoRNA' in x['predicted_sc_labels'] else 'snoRNA' if 'SNOR' in x['predicted_sc_labels'] else 'protein_coding' if 'RPL37A' in x['predicted_sc_labels'] else 'lncRNA' if 'SNHG1' in x['predicted_sc_labels'] else x['predicted_sc_labels'], axis=1))
    #add mc classes
    df = df.assign(actual_mc_labels=curr_mc_classes_df[0].values.tolist())
    #add sc classes
    df = df.assign(actual_sc_labels=curr_sc_classes_df[0].values.tolist())
    #compute accuracy
    df = df.assign(mc_accuracy=df.apply(lambda x: 1 if x['actual_mc_labels'] == x['predicted_mc_labels'] else 0, axis=1))
    df = df.assign(sc_accuracy=df.apply(lambda x: 1 if x['actual_sc_labels'] == x['predicted_sc_labels'] else 0, axis=1))

    #use plotly to plot confusion matrix based on mc classes
    mc_confusion_matrix = df.groupby(['actual_mc_labels','predicted_mc_labels'])['mc_accuracy'].count().unstack()
    mc_confusion_matrix = mc_confusion_matrix.fillna(0)
    mc_confusion_matrix = mc_confusion_matrix.apply(lambda x: x/x.sum(), axis=1)
    mc_confusion_matrix = mc_confusion_matrix.applymap(lambda x: round(x,2))
    #for columns not in rows, sum the column values and add them to a new column called 'other'
    other_col = [0]*mc_confusion_matrix.shape[0]
    for i in [i for i in mc_confusion_matrix.columns if i not in mc_confusion_matrix.index.tolist()]:
        other_col += mc_confusion_matrix[i]
    mc_confusion_matrix['other'] = other_col
    #add an other row with all zeros
    mc_confusion_matrix.loc['other'] = [0]*mc_confusion_matrix.shape[1]
    #drop all columns not in rows
    mc_confusion_matrix = mc_confusion_matrix.drop([i for i in mc_confusion_matrix.columns if i not in mc_confusion_matrix.index.tolist()], axis=1)
    #plot confusion matri
    fig = go.Figure(data=go.Heatmap(
            z=mc_confusion_matrix.values,
            x=mc_confusion_matrix.columns,
            y=mc_confusion_matrix.index,
            hoverongaps = False))
    #add z values to heatmap
    for i in range(len(mc_confusion_matrix.index)):
        for j in range(len(mc_confusion_matrix.columns)):
            fig.add_annotation(text=str(mc_confusion_matrix.values[i][j]), x=mc_confusion_matrix.columns[j], y=mc_confusion_matrix.index[i],
                                showarrow=False, font_size=25, font_color='red')
    #add title
    fig.update_layout(title_text='Confusion matrix based on mc classes for false novel sequences')
    #label x axis and y axis
    fig.update_xaxes(title_text='Predicted mc class')
    fig.update_yaxes(title_text='Actual mc class')
    #save
    if save_figs:
        fig.write_image('bin/lc_figures/confusion_matrix_mc_classes_false_novel.png')
            
            
def compute_accuracy(prediction_pd,sc_classes_df,mc_classes_df,seperate_outliers = False,fig_prefix:str = '',save_figs:bool=False):
    font_size = 25
    if fig_prefix == 'LC-familiar':
        font_size = 10
    #rename Labels to predicted_sc_labels
    prediction_pd = prediction_pd.rename(columns={'Labels':'predicted_sc_labels'})

    for model in prediction_pd['Model'].unique():
        #get model predictions
        num_rows = sc_classes_df.shape[0]
        model_prediction_pd = prediction_pd[prediction_pd['Model'] == model]
        model_prediction_pd = model_prediction_pd.set_index('Sequence')
        #filter index of model_prediction_pd to contain indices of sc_classes_df
        model_prediction_pd = model_prediction_pd.loc[[i for i in model_prediction_pd.index if i in sc_classes_df.index]]

        try: #try because ensemble models do not have a folder
            #check how many of the hico seqs exist in the train_df
            embedds_path = f'/media/ftp_share/hbdx/analysis/tcga/TransfoRNA_I_FULL_V4/sub_class/{model}/embedds/'
            results:Results_Handler = Results_Handler(path=embedds_path,splits=['train'])
        except:
            embedds_path = f'/media/ftp_share/hbdx/analysis/tcga/TransfoRNA_I_FULL_V4/sub_class/Seq-Rev/embedds/'
            results:Results_Handler = Results_Handler(path=embedds_path,splits=['train'])
            
        train_seqs = set(results.splits_df_dict['train_df']['RNA Sequences']['0'].values.tolist())
        common_seqs = train_seqs.intersection(set(model_prediction_pd.index.tolist()))
        print(f'Number of common seqs between train_df and {model} is {len(common_seqs)}')
        #print(f'removing overlaping sequences between train set and inference')
        #remove common_seqs from model_prediction_pd
        #model_prediction_pd = model_prediction_pd.drop(common_seqs)


        #compute number of sequences where NLD is higher than Novelty Threshold
        num_outliers = sum(model_prediction_pd['NLD'] > model_prediction_pd['Novelty Threshold'])
        false_novel_df = model_prediction_pd[model_prediction_pd['NLD'] > model_prediction_pd['Novelty Threshold']]

        plot_confusion_false_novel(false_novel_df,sc_classes_df,mc_classes_df,save_figs)
        #draw a pie chart depicting number of outliers per actual_mc_labels
        fig_outl = mc_classes_df.loc[false_novel_df.index][0].value_counts().plot.pie(autopct='%1.1f%%',figsize=(6, 6))
        fig_outl.set_title(f'False Novel per MC for {model}: {num_outliers}')
        if save_figs:
            fig_outl.get_figure().savefig(f'bin/lc_figures/false_novel_mc_{model}.png')
            fig_outl.get_figure().clf()
        #get number of unique sub classes per major class in false_novel_df
        false_novel_sc_freq_df = sc_classes_df.loc[false_novel_df.index][0].value_counts().to_frame()
        #save index as csv
        #false_novel_sc_freq_df.to_csv(f'false_novel_sc_freq_df_{model}.csv')
        #add mc to false_novel_sc_freq_df
        false_novel_sc_freq_df['MC'] = false_novel_sc_freq_df.index.map(lambda x: sc_to_mc_mapper_dict[x])
        #plot pie chart showing unique sub classes per major class in false_novel_df
        fig_outl_sc = false_novel_sc_freq_df.groupby('MC')[0].sum().plot.pie(autopct='%1.1f%%',figsize=(6, 6))
        fig_outl_sc.set_title(f'False novel: No. Unique sub classes per MC {model}: {num_outliers}')
        if save_figs:
            fig_outl_sc.get_figure().savefig(f'bin/lc_figures/{fig_prefix}_false_novel_sc_{model}.png')
            fig_outl_sc.get_figure().clf()
            #filter outliers
        if seperate_outliers:
            model_prediction_pd = model_prediction_pd[model_prediction_pd['NLD'] <= model_prediction_pd['Novelty Threshold']]
        else:
            #set the predictions of outliers to 'Outlier'
            model_prediction_pd.loc[model_prediction_pd['NLD'] > model_prediction_pd['Novelty Threshold'],'predicted_sc_labels'] = 'Outlier'
            model_prediction_pd.loc[model_prediction_pd['NLD'] > model_prediction_pd['Novelty Threshold'],'predicted_mc_labels'] = 'Outlier'
            sc_to_mc_mapper_dict['Outlier'] = 'Outlier'

        #filter index of sc_classes_df to contain indices of model_prediction_pd
        curr_sc_classes_df = sc_classes_df.loc[[i for i in model_prediction_pd.index if i in sc_classes_df.index]]
        curr_mc_classes_df = mc_classes_df.loc[[i for i in model_prediction_pd.index if i in mc_classes_df.index]]
        #convert Labels to mc_Labels
        model_prediction_pd = model_prediction_pd.assign(predicted_mc_labels=model_prediction_pd.apply(lambda x: sc_to_mc_mapper_dict[x['predicted_sc_labels']] if x['predicted_sc_labels'] in sc_to_mc_mapper_dict else 'miRNA' if 'miR' in x['predicted_sc_labels'] else 'tRNA' if 'tRNA' in x['predicted_sc_labels'] else 'rRNA' if 'rRNA' in x['predicted_sc_labels'] else 'snRNA' if 'snRNA' in x['predicted_sc_labels'] else 'snoRNA' if 'snoRNA' in x['predicted_sc_labels'] else 'snoRNA' if 'SNOR' in x['predicted_sc_labels'] else 'protein_coding' if 'RPL37A' in x['predicted_sc_labels'] else 'lncRNA' if 'SNHG1' in x['predicted_sc_labels'] else x['predicted_sc_labels'], axis=1))
        #add mc classes
        model_prediction_pd = model_prediction_pd.assign(actual_mc_labels=curr_mc_classes_df[0].values.tolist())
        #add sc classes
        model_prediction_pd = model_prediction_pd.assign(actual_sc_labels=curr_sc_classes_df[0].values.tolist())
        #correct labels
        model_prediction_pd['predicted_sc_labels'] = correct_labels(model_prediction_pd['predicted_sc_labels'],model_prediction_pd['actual_sc_labels'],sc_to_mc_mapper_dict)
        #compute accuracy
        model_prediction_pd = model_prediction_pd.assign(mc_accuracy=model_prediction_pd.apply(lambda x: 1 if x['actual_mc_labels'] == x['predicted_mc_labels'] else 0, axis=1))
        model_prediction_pd = model_prediction_pd.assign(sc_accuracy=model_prediction_pd.apply(lambda x: 1 if x['actual_sc_labels'] == x['predicted_sc_labels'] else 0, axis=1))
            
        if not seperate_outliers:
            cols_to_save = ['actual_mc_labels','predicted_mc_labels','predicted_sc_labels','actual_sc_labels']
            total_false_mc_predictions_df = model_prediction_pd[model_prediction_pd.actual_mc_labels != model_prediction_pd.predicted_mc_labels].loc[:,cols_to_save]
            #add a column indicating if NLD is greater than Novelty Threshold
            total_false_mc_predictions_df['is_novel'] = model_prediction_pd.loc[total_false_mc_predictions_df.index]['NLD'] > model_prediction_pd.loc[total_false_mc_predictions_df.index]['Novelty Threshold']
            #save
            total_false_mc_predictions_df.to_csv(f'bin/lc_files/{fig_prefix}_total_false_mcs_w_out_{model}.csv')
            total_true_mc_predictions_df = model_prediction_pd[model_prediction_pd.actual_mc_labels == model_prediction_pd.predicted_mc_labels].loc[:,cols_to_save]
            #add a column indicating if NLD is greater than Novelty Threshold
            total_true_mc_predictions_df['is_novel'] = model_prediction_pd.loc[total_true_mc_predictions_df.index]['NLD'] > model_prediction_pd.loc[total_true_mc_predictions_df.index]['Novelty Threshold']
            #save
            total_true_mc_predictions_df.to_csv(f'bin/lc_files/{fig_prefix}_total_true_mcs_w_out_{model}.csv')

        print('Model: ', model)
        print('num_outliers: ', num_outliers)
        #print accuracy including outliers
        print('mc_accuracy: ', model_prediction_pd['mc_accuracy'].mean())
        print('sc_accuracy: ', model_prediction_pd['sc_accuracy'].mean())
        
        #print balanced accuracy
        print('mc_balanced_accuracy: ', model_prediction_pd.groupby('actual_mc_labels')['mc_accuracy'].mean().mean())
        print('sc_balanced_accuracy: ', model_prediction_pd.groupby('actual_sc_labels')['sc_accuracy'].mean().mean())

        #use plotly to plot confusion matrix based on mc classes
        mc_confusion_matrix = model_prediction_pd.groupby(['actual_mc_labels','predicted_mc_labels'])['mc_accuracy'].count().unstack()
        mc_confusion_matrix = mc_confusion_matrix.fillna(0)
        mc_confusion_matrix = mc_confusion_matrix.apply(lambda x: x/x.sum(), axis=1)
        mc_confusion_matrix = mc_confusion_matrix.applymap(lambda x: round(x,4))
        #for columns not in rows, sum the column values and add them to a new column called 'other'
        other_col = [0]*mc_confusion_matrix.shape[0]
        for i in [i for i in mc_confusion_matrix.columns if i not in mc_confusion_matrix.index.tolist()]:
            other_col += mc_confusion_matrix[i]
        mc_confusion_matrix['other'] = other_col
        #add an other row with all zeros
        mc_confusion_matrix.loc['other'] = [0]*mc_confusion_matrix.shape[1]
        #drop all columns not in rows
        mc_confusion_matrix = mc_confusion_matrix.drop([i for i in mc_confusion_matrix.columns if i not in mc_confusion_matrix.index.tolist()], axis=1)
        #plot confusion matrix

        fig = go.Figure(data=go.Heatmap(
                    z=mc_confusion_matrix.values,
                    x=mc_confusion_matrix.columns,
                    y=mc_confusion_matrix.index,
                    colorscale='Blues',
                    hoverongaps = False))
        #add z values to heatmap
        for i in range(len(mc_confusion_matrix.index)):
            for j in range(len(mc_confusion_matrix.columns)):
                fig.add_annotation(text=str(round(mc_confusion_matrix.values[i][j],2)), x=mc_confusion_matrix.columns[j], y=mc_confusion_matrix.index[i],
                                    showarrow=False, font_size=font_size, font_color='black')

        fig.update_layout(
            title='Confusion matrix for mc classes - ' + model + ' - ' + 'mc B. Acc: ' + str(round(model_prediction_pd.groupby('actual_mc_labels')['mc_accuracy'].mean().mean(),2)) \
                + ' - ' + 'sc B. Acc: ' + str(round(model_prediction_pd.groupby('actual_sc_labels')['sc_accuracy'].mean().mean(),2)) + '<br>' + \
                    'percent false novel: ' + str(round(num_outliers/num_rows,2)),
            xaxis_nticks=36)
        #label x axis and y axis
        fig.update_xaxes(title_text='Predicted mc class')
        fig.update_yaxes(title_text='Actual mc class')
        if save_figs:
            #save plot
            if seperate_outliers:
                fig.write_image(f'bin/lc_figures/{fig_prefix}_LC_confusion_matrix_mc_no_out_' + model + '.png')
                #save svg
                fig.write_image(f'bin/lc_figures/{fig_prefix}_LC_confusion_matrix_mc_no_out_' + model + '.svg')
            else:
                fig.write_image(f'bin/lc_figures/{fig_prefix}_LC_confusion_matrix_mc_outliers_' + model + '.png')
                #save svg
                fig.write_image(f'bin/lc_figures/{fig_prefix}_LC_confusion_matrix_mc_outliers_' + model + '.svg')
        print('\n')


if __name__ == '__main__':
    #####################################################################################################################
    mapping_dict_path = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA//data/subclass_to_annotation.json'
    LC_path = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v02/LC__ngs__DI_HB_GEL-23.1.2.h5ad'
    #LC_path = '/media/ftp_share/hbdx/analysis/data/LC_DI_HB_GEL/LC__ngs__DI_HB_GEL-23.1.1.h5ad'
    #LC_path = '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v02/TCGA__ngs__miRNA_log2RPM-23.4.0.h5ad'
    
    
    trained_on = 'full' #id or full
    save_figs = True
    
    infer_aa = infer_relaxed_mirna = infer_hico = infer_ood = infer_other_affixes = infer_random = infer_fused = infer_na = infer_loco = False

    split = 'infer_relaxed_mirna'#sys.argv[1]
    print(f'Running inference for {split}')
    if split == 'infer_aa':
        infer_aa = True
    elif split == 'infer_relaxed_mirna':
        infer_relaxed_mirna = True
    elif split == 'infer_hico':
        infer_hico = True
    elif split == 'infer_ood':
        infer_ood = True
    elif split == 'infer_other_affixes':
        infer_other_affixes = True
    elif split == 'infer_random':
        infer_random = True
    elif split == 'infer_fused':
        infer_fused = True
    elif split == 'infer_na':
        infer_na = True
    elif split == 'infer_loco':
        infer_loco = True

    #####################################################################################################################
    #only one of infer_aa or infer_relaxed_mirna or infer_normal or infer_ood or infer_hico should be true
    if sum([infer_aa,infer_relaxed_mirna,infer_hico,infer_ood,infer_other_affixes,infer_random,infer_fused,infer_na,infer_loco]) != 1:
        raise Exception('Only one of infer_aa or infer_relaxed_mirna or infer_normal or infer_ood or infer_hico or infer_other_affixes or infer_random or infer_fused or infer_na should be true')

    #set fig_prefix
    if infer_aa:
        fig_prefix = '5\'A-affixes'
    elif infer_other_affixes:
        fig_prefix = 'other_affixes'
    elif infer_relaxed_mirna:
        fig_prefix = 'Relaxed-miRNA'
    elif infer_hico:
        fig_prefix = 'LC-familiar'
    elif infer_ood:
        fig_prefix = 'LC-novel'
    elif infer_random:
        fig_prefix = 'Random'
    elif infer_fused:
        fig_prefix = 'Fused'
    elif infer_na:
        fig_prefix = 'NA'
    elif infer_loco:
        fig_prefix = 'LOCO'

    ad = load(LC_path)
    sc_to_mc_mapper_dict = load(mapping_dict_path)
    #select hico sequences
    hico_seqs = ad.var.index[ad.var['hico']].tolist()
    art_affix_seqs = ad.var[~ad.var['five_prime_adapter_filter']].index.tolist()
    
    if infer_hico:
        hico_seqs = hico_seqs

    if infer_aa:
        hico_seqs = art_affix_seqs

    if infer_other_affixes:
        hico_seqs = ad.var[~ad.var['hbdx_spikein_affix_filter']].index.tolist()
    
    if infer_na:
        hico_seqs = ad.var[ad.var.subclass_name == 'no_annotation'].index.tolist()
    
    if infer_loco:
        hico_seqs = ad.var[~ad.var['hico']][ad.var.subclass_name != 'no_annotation'].index.tolist()

    #for mirnas
    if infer_relaxed_mirna:
        #subclass name must contain miR, let, Let and not contain ; and that are not hico
        mirnas_seqs = ad.var[ad.var.subclass_name.str.contains('miR') | ad.var.subclass_name.str.contains('let')][~ad.var.subclass_name.str.contains(';')].index.tolist()
        #remove the ones that are true in ad.var.hico column
        hico_seqs = list(set(mirnas_seqs).difference(set(hico_seqs)))

        #novel mirnas
        #mirnas_not_in_train_mask = (ad.var['hico']==True).values *  ~(ad.var['subclass_name'].isin(mirna_train_sc)).values * (ad.var['small_RNA_class_annotation'].isin(['miRNA']))
        #hicos = ad.var[mirnas_not_in_train_mask].index.tolist()

    
    if infer_random:
        #create random sequences
        random_seqs = []
        while len(random_seqs) < 200:
            random_seq = ''.join(random.choices(['A','C','G','T'], k=randint(18,30)))
            if random_seq not in random_seqs:
                random_seqs.append(random_seq)
        hico_seqs = random_seqs
    
    if infer_fused:
        hico_seqs = get_fused_seqs(hico_seqs,num_sequences=200)
    
    
    #hico_seqs = ad.var[ad.var.subclass_name.str.contains('mir')][~ad.var.subclass_name.str.contains(';')]['subclass_name'].index.tolist()
    hico_seqs = [seq for seq in hico_seqs if len(seq) <= 30]  
     

    #run prediction
    prediction_pd = predict_transforna_all_models(hico_seqs,trained_on=trained_on)
    prediction_pd['split'] = fig_prefix
    #the if condition here is to make sure to filter seqs with sub classes not used in training
    if not infer_ood and not infer_relaxed_mirna and not infer_hico:
        prediction_pd.to_csv(f'bin/lc_files/{fig_prefix}_lev_dist_df.csv')
    if infer_aa or infer_other_affixes or infer_random or infer_fused:
        for model in prediction_pd.Model.unique():
            num_non_novel = sum(prediction_pd[prediction_pd.Model == model]['Is Familiar?'])
            print(f'Number of non novel sequences for {model} is {num_non_novel}')
            print(f'Percent non novel for {model} is {num_non_novel/len(prediction_pd[prediction_pd.Model == model])}, the lower the better')
    
    else:  
        if infer_na or infer_loco:
            #print number of Is Familiar per model
            for model in prediction_pd.Model.unique():
                num_non_novel = sum(prediction_pd[prediction_pd.Model == model]['Is Familiar?'])
                print(f'Number of non novel sequences for {model} is {num_non_novel}')
                print(f'Percent non novel for {model} is {num_non_novel/len(prediction_pd[prediction_pd.Model == model])}, the higher the better')
                print('\n')
        else:  
            #only to get classes used for training
            prediction_single_pd = predict_transforna(hico_seqs[0],model='Seq',logits_flag = True,trained_on=trained_on)
            sub_classes_used_for_training = prediction_single_pd.columns.tolist()
        

            mc_classes_df,sc_classes_df = get_mc_sc(ad,hico_seqs,sub_classes_used_for_training,sc_to_mc_mapper_dict,ood_flag=infer_ood)
            if infer_ood:
                for model in prediction_pd.Model.unique():
                    #filter sequences in prediction_pd to only include sequences in sc_classes_df
                    curr_prediction_pd = prediction_pd[prediction_pd['Sequence'].isin(sc_classes_df.index)]
                    #filter curr_prediction toonly include model
                    curr_prediction_pd = curr_prediction_pd[curr_prediction_pd.Model == model]
                    num_seqs = curr_prediction_pd.shape[0]
                    #filter Is Familiar
                    curr_prediction_pd = curr_prediction_pd[curr_prediction_pd['Is Familiar?']]
                    #filter sc_classes_df to only include sequences in curr_prediction_pd
                    curr_sc_classes_df = sc_classes_df[sc_classes_df.index.isin(curr_prediction_pd['Sequence'].values)]
                    #correct labels and remove the correct labels from the curr_prediction_pd
                    curr_prediction_pd['Net-Label'] = correct_labels(curr_prediction_pd['Net-Label'].values,curr_sc_classes_df[0].values,sc_to_mc_mapper_dict)
                    #filter rows in curr_prediction where Labels is equal to sc_classes_df[0]
                    curr_prediction_pd = curr_prediction_pd[curr_prediction_pd['Net-Label'].values != curr_sc_classes_df[0].values]
                    num_non_novel = len(curr_prediction_pd)
                    print(f'Number of non novel sequences for {model} is {num_non_novel}')
                    print(f'Percent non novel for {model} is {num_non_novel/num_seqs}, the lower the better')
                    print('\n')
            else:
                #filter prediction_pd to include only sequences in prediction_pd
                
                #compute_accuracy(prediction_pd,sc_classes_df,mc_classes_df,seperate_outliers=False,fig_prefix = fig_prefix,save_figs=save_figs)
                compute_accuracy(prediction_pd,sc_classes_df,mc_classes_df,seperate_outliers=True,fig_prefix = fig_prefix,save_figs=save_figs)

            if infer_ood or infer_relaxed_mirna or infer_hico:
                prediction_pd = prediction_pd[prediction_pd['Sequence'].isin(sc_classes_df.index)]
                #save lev_dist_df
                prediction_pd.to_csv(f'bin/lc_files/{fig_prefix}_lev_dist_df.csv')



        