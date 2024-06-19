#%%
import pandas as pd
from transforna import load,get_fused_seqs,predict_transforna_all_models
import os
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
from transforna import fold_sequences
from difflib import SequenceMatcher
from typing import List

df_general = pd.read_csv('/media/ftp_share/hbdx/analysis/tcga/transforna_dash_data/lc_tcga_summary_table_unique_2_df.csv')
df_umap = load('/media/ftp_share/hbdx/analysis/tcga/transforna_dash_data/lc_tcga_model_predictions_df.csv')
#%%
#####functions########
def longest_match(seqA, seqB):
    matches = []
    while True:
        match = SequenceMatcher(None, seqA, seqB).find_longest_match(0, len(seqA), 0, len(seqB))
        if match.size <= 1:
            break
        matches.append(seqA[match.a: match.a + match.size])
        #seqA should be include before the match and after the match
        seqA =  seqA[:match.a] +'K'+seqA[match.a + match.size:]
        seqB = seqB[:match.b]+ 'B'+seqB[match.b + match.size:]
    return matches

def get_ordered_longest_matches(reference_seqs:List[str], comparable_seqs:List[str],max_matches=2):
    '''
    returns a list of lists of the longest matches between the reference and comparable sequences
    If the longest match is not in the same order in the comparable sequence, the shortest match is reversed
    Input:
        reference_seqs: list of sequences taken as reference
        comparable_seqs: list of sequences to compare against
        max_matches: maximum number of matches to return
    '''
    longest_matches = [longest_match(reference_seqs[i], comparable_seqs[i]) for i in range(len(comparable_seqs))]

    #for list of long matches keep the max_matches longest matches
    longest_matches = [longest_matches[i][:max_matches] for i in range(len(longest_matches))]
    #print(longest_matches)
    #for each longest match list in the list of lists of longest matches, order the list by order of appearance in the feature
    sorted_matches =  [sorted(longest_matches[i], key=lambda x: comparable_seqs[i].find(x)) for i in range(len(longest_matches))]


    return sorted_matches

def get_ids_of_matches_in_both_sets(matches_list: List[List[str]], reference:str, comparable_seqs:List[str]):
    #longest_matches is a list of lists. each list contains at most two strings representing the longest match between the feature and the similar sequence
    #check if the same order also appears in the similar sequence. if not, reverse the shortest match
    ids_of_match_in_reference_seq = []
    ids_of_match_in_comparable_seqs = []
    for i in range(len(matches_list)):
        if len(matches_list[i]) == 2:
            if reference.find(matches_list[i][0]) > reference.find(matches_list[i][1]):
                matches_list[i] = matches_list[i][:1]
                ids_of_match_in_reference_seq.append([reference.find(matches_list[i][0])])
                ids_of_match_in_comparable_seqs.append([comparable_seqs[i].find(matches_list[i][0])]) 
            elif matches_list[i][0] in matches_list[i][1] or matches_list[i][1] in matches_list[i][0]:
                #ids appended should not be the same, therefore the find function should search in reference[find(matches_list[i][0])+len(matches_list[i][0]):]
                first_ref_match_idx = reference.find(matches_list[i][0])
                first_compable_match_idx = comparable_seqs[i].find(matches_list[i][0])
                ids_of_match_in_reference_seq.append([first_ref_match_idx, reference[first_ref_match_idx+len(matches_list[i][0]):].find(matches_list[i][1])+first_ref_match_idx+len(matches_list[i][0])])
                ids_of_match_in_comparable_seqs.append([first_compable_match_idx, comparable_seqs[i][first_compable_match_idx+len(matches_list[i][0]):].find(matches_list[i][1])+first_compable_match_idx+len(matches_list[i][0])])
            else:
                ids_of_match_in_reference_seq.append([reference.find(matches_list[i][0]), reference.find(matches_list[i][1])])
                ids_of_match_in_comparable_seqs.append([comparable_seqs[i].find(matches_list[i][0]), comparable_seqs[i].find(matches_list[i][1])])
        
        else:
            ids_of_match_in_reference_seq.append([reference.find(matches_list[i][0])])
            ids_of_match_in_comparable_seqs.append([comparable_seqs[i].find(matches_list[i][0])]) 

    return ids_of_match_in_reference_seq, ids_of_match_in_comparable_seqs, matches_list

def color_match(input_feat,idx_start, idx_end, color):
    return f'<span style="background-color:{color}">{input_feat[idx_start:idx_end]}</span>'

def color_matches(red_mask: List[bool], ids_of_match_in_comparable_seqs:List[List[int]], comparable_seqs:List[str], matches_list:List[List[str]]):
    #now loop over the mask and color the matches
    for i in range(len(red_mask)):
        #get first part of the sequence until the second match +  the first match (in green)
        before_first_match = comparable_seqs[i][:ids_of_match_in_comparable_seqs[i][0]]
        first_match = color_match(comparable_seqs[i],ids_of_match_in_comparable_seqs[i][0],ids_of_match_in_comparable_seqs[i][0]+len(matches_list[i][0]),'lightgreen')
        #check if there is a second match
        if len(ids_of_match_in_comparable_seqs[i]) == 2:
            #color the second match in green and get the part of the sequence after the second match until the end of the comparable sequence
            second_match = color_match(comparable_seqs[i],ids_of_match_in_comparable_seqs[i][1],ids_of_match_in_comparable_seqs[i][1]+len(matches_list[i][1]),'lightgreen')
            after_second_match = comparable_seqs[i][ids_of_match_in_comparable_seqs[i][1]+len(matches_list[i][1]):]

            #if mismatch between first and second match have the same length, color them in red
            if red_mask[i]:
                middle_mismatch = color_match(comparable_seqs[i],ids_of_match_in_comparable_seqs[i][0] + len(matches_list[i][0]),ids_of_match_in_comparable_seqs[i][1],'red')
            else:
                middle_mismatch = comparable_seqs[i][ids_of_match_in_comparable_seqs[i][0] + len(matches_list[i][0]):ids_of_match_in_comparable_seqs[i][1]]
                
            colored_seq = before_first_match + first_match + middle_mismatch + second_match + after_second_match
        
        else:
            #in case there is only one match, get the rest of the sequnece(after the first match)
            after_first_match = comparable_seqs[i][ids_of_match_in_comparable_seqs[i][0]+len(matches_list[i][0]):]
            colored_seq = before_first_match + first_match + after_first_match
    return colored_seq

def find_and_color_matches(reference_seq:str,comparable_seq:str):
    reference_seq_list = [reference_seq]
    comparable_seq_list = [comparable_seq]


    #pass the two lists to longest_match function to get the longest match between each pair
    longest_matches = get_ordered_longest_matches(reference_seq_list, comparable_seq_list)

    #longest_matches is a list of lists. each list contains at most two strings representing the longest match between the feature and the similar sequence
    #check if the same order also appears in the similar sequence. if not, reverse the shortest match
    ids_of_match_in_reference_seq, ids_of_match_in_comparable_seqs, longest_matches = get_ids_of_matches_in_both_sets(longest_matches, reference_seq_list[0], comparable_seq_list)
    #print(f'ids_of_match_in_reference_seq: {ids_of_match_in_reference_seq}')
    #print(f'ids_of_match_in_comparable_seqs: {ids_of_match_in_comparable_seqs}')
    #the folowing line is specifically for low complex sequences AAAAAAA...AAAAA for example.
    #edit exact_mismatch_length_mask such that if feat_id[0] == feat_id[1] then mask is true if longest_matches[i][1] can be found at index feat_id[0]+sim_id[1]
    exact_mismatch_length_mask = [True if len(feat_id) == 1 \
        #true if feat_id[0] - feat_id[1] == sim_id[0] - sim_id[1]
        else (True if feat_id[0] - feat_id[1] == sim_id[0] - sim_id[1] \
        #condition for low complex seqs
        else True if reference_seq[feat_id[0]:feat_id[0]+len(longest_matches[i][0])] == comparable_seq[sim_id[0]:sim_id[0]+len(longest_matches[i][0])] and reference_seq[feat_id[1]:feat_id[1]+len(longest_matches[i][1])] == comparable_seq[sim_id[1]:sim_id[1]+len(longest_matches[i][1])] \

        else False) for i,(feat_id, sim_id) in enumerate(zip(ids_of_match_in_reference_seq, ids_of_match_in_comparable_seqs))]
    
    #for all matches if the mask is true, change each list in ids_of_match_in_comparable_seqs to contain the highest difference (in ids_of_match_in_reference_seq or ids_of_match_in_comparable_seqs) in length 
    ids_of_match_in_comparable_seqs = [sim_id if not mask \
        else sim_id if len(sim_id) == 1 \
        else [sim_id[0],sim_id[0]+ids_of_match_in_reference_seq[i][1] - ids_of_match_in_reference_seq[i][0]] if  sim_id[1]-sim_id[0] < ids_of_match_in_reference_seq[i][1] - ids_of_match_in_reference_seq[i][0] \
        else sim_id
        for i,(mask,sim_id) in enumerate(zip(exact_mismatch_length_mask,ids_of_match_in_comparable_seqs))]
        #special check for 
    #print(exact_mismatch_length_mask)
    #both matches should be colored in green. after the first match ends and before the second match starts should be colored in red if and only if the mask is true
    #create a function that takes sequence idx_start, idx_end and color
    colored_seq = color_matches(exact_mismatch_length_mask,ids_of_match_in_comparable_seqs,comparable_seq_list,longest_matches)
    return colored_seq

def append_umap(pred_df,umap_df,umap_prefix:str=''):
    pred_df[f'{umap_prefix}UMAP1'] = 0
    pred_df[f'{umap_prefix}UMAP2'] = 0

    #for each model in mirna_df, get the umap coordinates from mirna_umap_df
    for model in umap_df.Model.unique():
        pred_df.loc[pred_df.Model == model,f'{umap_prefix}UMAP1'] = umap_df[umap_df.Model == model].UMAP1.values
        pred_df.loc[pred_df.Model == model,f'{umap_prefix}UMAP2'] = umap_df[umap_df.Model == model].UMAP2.values
    return pred_df

def append_novelty_viz_col(df:pd.DataFrame) -> pd.DataFrame:
    div = df['div']
    df = df.drop(columns=['div'])
    div[div > 2] = 2
    #map each value in div from 1 to 11
    div = 1 + (div * 5).astype(int)

    df["NLD vs Novelty"] = '>-----|-----<'
    for i in range(len(df)):
        char_list = list(df["NLD vs Novelty"][i])
        if int(div[i]) <6:
            charac = "✓"
        elif int(div[i]) >=7:
            charac = "✗"
        else:
            charac = "?"
        char_list[int(div[i])] = charac
        df["NLD vs Novelty"][i] = ''.join(char_list)
    
    #display table
    df["NLD vs Novelty"] = df["NLD vs Novelty"].apply(lambda x: f'<span style="color:green">{x[:6]}</span>'+f'<span style="color:goldenrod">{x[6:7]}</span>'+f'<span style="color:red">{x[7:]}</span>')
    return df

def append_models_agreeing_with_ensemble(ensemble_df:pd.DataFrame,all_models_df:pd.DataFrame) -> pd.DataFrame:
    '''
    ensemble df contains the predictions of the ensemble model while all_models_df contains the predictions of all models.
    this function appends a column that contains the models that agree with the ensemble model per query sequence.
    '''
    #if Is Familiar? is True, then get the models that have Is Familiar? == True and that have the same label as the ensemble model
    #if Is Familiar? is False, then get the models that have Is Familiar? == False regardless of the label
    #reset index
    print("computing models agreeing with ensemble...")
    ensemble_df = ensemble_df.reset_index(drop=True)
    ensemble_df['Models Agreeing on familiarity'] = ensemble_df.apply(lambda x: all_models_df[(all_models_df['Sequence'] == x['Sequence']) & \
        (all_models_df['Is Familiar?'] == x['Is Familiar?'])]['Model'].values.tolist(), axis=1)

    #convert the models to a string separated by ', '
    ensemble_df['Models Agreeing on familiarity'] = ensemble_df['Models Agreeing on familiarity'].apply(lambda x: ', '.join(x))
    print("coloring models agreeing with ensemble...")
    #color models in green if Is Familiar == True otherwise color them in red
    ensemble_df['Models Agreeing on familiarity'] = ensemble_df.apply(lambda x: f'<span style="color:green">{x["Models Agreeing on familiarity"]}</span>' if x['Is Familiar?'] else f'<span style="color:red">{x["Models Agreeing on familiarity"]}</span>', axis=1)

    return ensemble_df

def get_ensemble_table(inference_df):
    #get unique models
    inference_df = inference_df.sort_values(["Sequence", "Model"])
    inference_df['div'] = inference_df['NLD']/inference_df['Novelty Threshold']
    # for each sequence, choose the best model
    print('Choosing best model per sequence...')
    ensemble_df = append_models_agreeing_with_ensemble(inference_df[inference_df.Model == 'Ensemble'], inference_df[inference_df.Model != 'Ensemble'])
    print('appending novelty viz col...')
    ensemble_df = append_novelty_viz_col(ensemble_df)
    # rename values in 'Labels' column using mapping_dict_bins to get bin_pos
    #ensemble_df['Major Class'] = ensemble_df['Net-Label'].map(mapping_dict)
    #ensemble_df['Net-Label'] = ensemble_df['Net-Label'].map(mapping_dict_bins)
    #ensemble_df['kba'] = ensemble_df['kba'].map(mapping_dict_bins)
    return ensemble_df

def get_model_matching_lv_distance(seq:str,ensemble_df:pd.DataFrame,pred_df:pd.DataFrame,sim_df:pd.DataFrame):
    ensemble_lv = ensemble_df[ensemble_df.Sequence == seq]['NLD'].values[0]
    lv_dists =  pred_df[pred_df.Sequence == seq]['NLD'].values
    models = pred_df[pred_df.Sequence == seq]['Model'].values
    model_idx = list(lv_dists).index(ensemble_lv)
    model = models[model_idx]
    expl_seq = sim_df[(sim_df['Sequence'] == seq) & (sim_df.Model == model)]['Explanatory Sequence'].values[0]
    return expl_seq
######################
# %%
#get common columns
common_cols = list(set(df_general.columns).intersection(df_umap.columns))
tcga_path =  '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.04.0__var.csv'
mapping_dict_path = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA//data/subclass_to_annotation.json'
lc_path =  '/media/ftp_share/hbdx/annotation/feature_annotation/ANNOTATION/HBDxBase_annotation/TransfoRNA/compare_binning_strategies/v05/2024-04-19__230126_LC_DI_HB_GEL_v23.01.00/sRNA_anno_aggregated_on_seq.csv'
path_to_models = '/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/models/tcga/'
#read the test_embedds
scs_used_for_training =load('/nfs/home/yat_ldap/VS_Projects/TransfoRNA-Framework/models/tcga/TransfoRNA_FULL/sub_class/Seq/meta/hp_settings.yaml')['model_config']['class_mappings']
if False:#not os.path.exists('web_files/seqs_df.csv'):
    lc_df = pd.read_csv(lc_path)
    lc_df = lc_df[lc_df.sequence.str.len()<=30]
    tcga_df = pd.read_csv(tcga_path)
    mapping_dict = load(mapping_dict_path)
    #get sequences in a dataframe and a create two boolean columns LC and TCGA
    seqs_df = pd.concat([lc_df,tcga_df],axis=0)[['sequence','small_RNA_class_annotation','subclass_name','hico','five_prime_adapter_filter']]
    #rename small_RNA_class_annotation to Major RNA Class of KBA and subclass_name to KBA
    seqs_df.rename(columns={'small_RNA_class_annotation':'Major RNA Class of KBA','subclass_name':'KBA'},inplace=True)

    #add a column split which could be LOCO, HICO-Familiar, HICO-Novel, IsomiR, 5' Adapter or no_annotation
    #no_annotation: Split = no_annotation when KBA = no_annotation
    seqs_df['Split'] = 'N/A'
    seqs_df.loc[seqs_df['KBA']=='no_annotation','Split'] = 'no_annotation'
    #LOCO: Split = LOCO if hico = False and KBA != no_annotation
    seqs_df.loc[(seqs_df['hico']==False) & (seqs_df['KBA']!='no_annotation'),'Split'] = 'LOCO'
    #HICO-Familiar: Split = HICO-Familiar if hico = True and KBA is in scs_used_for_training
    seqs_df.loc[(seqs_df['hico']==True) & (seqs_df['KBA'].isin(scs_used_for_training)),'Split'] = 'HICO-Familiar'
    #HICO-Novel: Split = HICO-Novel if hico = True and KBA is not in scs_used_for_training
    seqs_df.loc[(seqs_df['hico']==True) & (~seqs_df['KBA'].isin(scs_used_for_training)),'Split'] = 'HICO-Novel'
    #IsomiR: Split = IsomiR if hico = False and KBA.str.contains('miR') | infer_df.KBA.str.contains('let')] and KBA should not contain ;
    seqs_df.loc[(seqs_df['hico']==False) & (seqs_df['KBA'].str.contains('miR') | seqs_df['KBA'].str.contains('let')) & (~seqs_df['KBA'].str.contains(';')),'Split'] = 'IsomiR'
    #5' Adapter: Split = 5' Adapter is hico = False and five_prime_adapter_filter == False
    seqs_df.loc[(seqs_df['hico']==False) & (seqs_df['five_prime_adapter_filter']==False),'Split'] = '5\' Adapter'


    #random seqs
    import random
    random_seqs = []
    #create random seqs with length from 18 to 30
    for i in range(200):
        random_seqs.append(''.join(random.choices(['A','C','G','T'],k=random.randint(18,30))))

    fused_seqs = get_fused_seqs(seqs_df[seqs_df['Split']=='HICO-Familiar'].sequence.values,num_sequences=200)

    random_df = pd.DataFrame({'sequence':random_seqs,'LC':False,'TCGA':False,'Split':'Random'})
    fused_df = pd.DataFrame({'sequence':fused_seqs,'LC':False,'TCGA':False,'Split':'Fused'})
    seqs_df = pd.concat([seqs_df,random_df,fused_df],axis=0).reset_index(drop=True)
    seqs_df['AD'] = seqs_df['Split'].isin(['Random','Fused'])

    seqs_df.set_index('sequence',inplace=True,drop=True)
    #drop duplicate indices
    seqs_df = seqs_df[~seqs_df.index.duplicated(keep='first')]
    seqs_df['TCGA'] = seqs_df.index.isin(tcga_df.sequence)
    seqs_df['LC'] = seqs_df.index.isin(lc_df.sequence)
    seqs_df['AD'] = seqs_df['Split'].isin(['Random','Fused'])
    #rename index to Sequence
    seqs_df.index.name = 'Sequence'
    #some sequences are in multiple datasets. make a column that contains list of datasets for weach sequence
    seqs_df['Datasets'] = seqs_df[['TCGA','LC','AD']].apply(lambda x: [x.index[i] for i in range(3) if x[i]],axis=1)
#else:
#    seqs_df = pd.read_csv('web_files/seqs_df.csv')
#    seqs_df.set_index('Sequence',inplace=True)
# %%


#predict similarity 
#if not os.path.exists('web_files/lc_tcga_similarity_df.csv'):    
#    print('predicting similarity')
#    sim_df = predict_transforna_all_models(seqs_df.index,path_to_models=path_to_models,similarity_flag=True)
#    sim_df.to_csv('web_files/lc_tcga_similarity_df.csv',index=False)
#else:
#    sim_df = pd.read_csv('web_files/lc_tcga_similarity_df.csv')

#create ensemble and per model predictions file
#if not os.path.exists('web_files/lc_tcga_model_predictions_df.csv'):
#    print('predicting labels')
#    pred_df = predict_transforna_all_models(seqs_df.index,path_to_models=path_to_models)
#
#    pred_df['Major RNA Class of Net-Label'] = pred_df['Net-Label'].map(mapping_dict)
#    pred_df = pred_df.join(seqs_df[['KBA','Major RNA Class of KBA','Split','LC','TCGA','AD','Datasets']],on='Sequence')
#
#    pred_df.to_csv('web_files/lc_tcga_model_predictions_df.csv',index=False)
#else:
#    pred_df = pd.read_csv('web_files/lc_tcga_model_predictions_df.csv')
#
#if not os.path.exists('web_files/lc_tcga_ensemble_predictions_df.csv'):
#    print('fetching ensemble table')
#    pred_df = pd.read_csv('web_files/lc_tcga_model_predictions_df.csv')
#    ensemble_df = get_ensemble_table(pred_df)
#    with Pool(mp.cpu_count()) as p:
#        ensemble_df['Explanatory Sequence'] = p.map(partial(get_model_matching_lv_distance,ensemble_df=ensemble_df,pred_df=pred_df,sim_df=sim_df),ensemble_df.Sequence)
#    ensemble_df.to_csv('web_files/lc_tcga_ensemble_predictions_df.csv',index=False)
#
#else:
#    ensemble_df = pd.read_csv('web_files/lc_tcga_ensemble_predictions_df.csv')

if not os.path.exists('web_files/lc_tcga_ensemble_final_predictions_colored_df.csv'):
    print('Augmenting ensemble table with secondary structure and colored explanatory sequence...')
    ensemble_df = pd.read_csv('web_files/lc_tcga_ensemble_final_predictions_df.csv')
    
    #get secondary structure of each sequence
    ensemble_df['Secondary Structure'] = fold_sequences(ensemble_df.Sequence.values,temperature=37)['structure_37'].values
    ensemble_df['Ensemble Final Prediction'] = ensemble_df.apply(lambda x: 'Novel' if x['Is Familiar?'] == False else x['Major RNA Class of Net-Label'],axis=1)

    #ensemble_df['Explanatory Sequence Colored'] = ensemble_df.apply(lambda row: find_and_color_matches(row.Sequence,row['Explanatory Sequence']),axis=1)
    with Pool(mp.cpu_count()) as p:
        ensemble_df['Explanatory Sequence Colored'] = p.starmap(find_and_color_matches, [(row.Sequence,row['Explanatory Sequence']) for idx,row in ensemble_df.iterrows()])
    ensemble_df.to_csv('web_files/lc_tcga_ensemble_final_predictions_colored_df.csv',index=False)
else:
    ensemble_df = pd.read_csv('web_files/lc_tcga_ensemble_final_predictions_colored_df.csv')

#get umap
#if not os.path.exists('web_files/sc_lc_tcga_umap_df.csv'):
#    print('predicting sc umap')
#    sc_umap_df = predict_transforna_all_models(seqs_df.index,path_to_models=path_to_models,umap_flag=True)
#    sc_umap_df.to_csv('web_files/lc_tcga_umap_df.csv',index=False)
#else:
#    sc_umap_df = pd.read_csv('web_files/sc_lc_tcga_umap_df.csv')
#
#if not os.path.exists('web_files/mc_lc_tcga_umap_df.csv'):
#    print('predicting mc umap')
#    mc_umap_df = predict_transforna_all_models(seqs_df.index,mc_or_sc='major_class',path_to_models=path_to_models,umap_flag=True)
#    mc_umap_df.to_csv('web_files/mc_lc_tcga_umap_df.csv',index=False)
#else:
#    mc_umap_df = pd.read_csv('web_files/mc_lc_tcga_umap_df.csv')
#
#pred_df = append_umap(pred_df,sc_umap_df,'SC_')
#pred_df = append_umap(pred_df,mc_umap_df,'MC_')
#
#pred_df['Major RNA Class of Net-Label after Novelty Prediction'] = pred_df.apply(lambda x: 'Novel' if x['Is Familiar?'] == False else x['Major RNA Class of Net-Label'],axis=1)
#pred_df.drop(columns=['Split', 'LC', 'TCGA', 'AD', 'Datasets'],inplace=True)
##save pred_df
#pred_df.to_csv('web_files/lc_tcga_model_predictions_final_df.csv',index=False)
# %%
