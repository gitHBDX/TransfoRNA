
from pathlib import Path
from typing import Dict

from ..processing.seq_tokenizer import SeqTokenizer
from ..utils.file import load
from ..utils.utils import *


def infer_tcga(cfg:Dict= None,path:str = None):
    if cfg['tensorboard']:
        from ..callbacks.tbWriter import writer
    cfg,net = get_model(cfg,path)
    inference_path = cfg['inference_settings']['sequences_path']
    original_infer_df = load(inference_path, index_col=0)
    predicted_labels,logits,_,_,all_data,max_len,net,infer_df = infer_from_pd(cfg,net,original_infer_df,SeqTokenizer)
    
    #create inference_output if it does not exist
    if not os.path.exists(f"inference_output"):
        os.makedirs(f"inference_output")
    if cfg['log_embedds']:
        embedds_pd = log_embedds(cfg,net,all_data['infere_rna_seq'])
        embedds_pd.to_csv(f"inference_output/{cfg['model_name']}_embedds.csv")
    
    prepare_inference_results_tcga(cfg,predicted_labels,logits,all_data,max_len)
    
    #if sequences were trimmed, add mapping of trimmed sequences to original sequences
    if original_infer_df.shape[0] != infer_df.shape[0]:
        all_data["infere_rna_seq"] = add_original_seqs_to_predictions(infer_df,all_data['infere_rna_seq'])              
    #save
    all_data["infere_rna_seq"].to_csv(f"inference_output/{cfg['model_name']}_inference_results.csv")

    if cfg['tensorboard']:
        writer.close()
    return predicted_labels