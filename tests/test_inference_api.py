#%%
from src.transforna.inference_api import predict_transforna_all_models
seqs = ["ATGCCCAAATTTGGGACTA","GGGGGGCCCCCCTTTTTTT"]
preds_df = predict_transforna_all_models(seqs)
# %%
