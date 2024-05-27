from transforna import predict_transforna, predict_transforna_all_models

seqs = [
'AACGAAGCTCGACTTTTAAGG',
'GTCCACCCCAAAGCGTAGG']

path_to_models = '/path/to/tcga/models/'
sc_preds_id_df = predict_transforna_all_models(seqs,path_to_models = path_to_models) #/models/tcga/
#%%
#get sc predictions for models trained on id (in distribution)
sc_preds_id_df = predict_transforna(seqs, model="seq",trained_on='id',path_to_models = path_to_models)
#get sc predictions for models trained on full (all sub classes)  
sc_preds_df = predict_transforna(seqs, model="seq",path_to_models = path_to_models)
#predict using models trained on major class
mc_preds_df = predict_transforna(seqs, model="seq",mc_or_sc='major_class',path_to_models = path_to_models)
#get logits
logits_df = predict_transforna(seqs, model="seq",logits_flag=True,path_to_models = path_to_models)
#get embedds
embedd_df = predict_transforna(seqs, model="seq",embedds_flag=True,path_to_models = path_to_models)
#get the top 4 similar sequences
sim_df = predict_transforna(seqs, model="seq",similarity_flag=True,n_sim=4,path_to_models = path_to_models)
#get umaps
umaps_df = predict_transforna(seqs, model="seq",umaps_flag=True,path_to_models = path_to_models)


all_preds_df = predict_transforna_all_models(seqs,path_to_models=path_to_models)
all_preds_df

# %%
