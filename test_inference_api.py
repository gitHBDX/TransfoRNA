#%%
from src.transforna.inference_api import predict_transforna,predict_transforna_all_models
seqs = ["ATGCCCAAATTTGGGACTA","GGGGGGCCCCCCTTTTTTT"]
preds_df = predict_transforna(seqs, model="seq")
preds_all_df = predict_transforna_all_models(seqs)

# %%
