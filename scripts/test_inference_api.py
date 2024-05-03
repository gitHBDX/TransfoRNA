#%%
from transforna import predict_transforna, predict_transforna_all_models

seqs = ["ATGCCCAAATTTGGGACTA","GGGGGGCCCCCCTTTTTTT"]
preds_df = predict_transforna(seqs, model="seq",path_to_id_models = 'models/tcga/')
preds_all_df = predict_transforna_all_models(seqs,path_to_id_models='models/tcga/')
preds_all_df

# %%
