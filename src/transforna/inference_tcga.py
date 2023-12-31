
from typing import Dict
from .utils.file import load
from .utils.utils import *
from .dataset.dataset_tcga import PrepareGeneData as DatasetTcga
from pathlib import Path

def infer_tcga(cfg:Dict= None,path:str = None):
    if cfg['tensorboard']:
        from .tbWriter import writer
    cfg,net = get_model(cfg,path)
    inference_path = Path(__file__).parent.parent.absolute()
    infer_pd = load(inference_path / cfg["inference_settings"]["sequences_path"],index_col=0)
    predicted_labels,logits,_,_,all_data,max_len,net = infer_from_pd(cfg,net,infer_pd,DatasetTcga)
    #create inference_output if it does not exist
    if not os.path.exists(f"{inference_path}/inference_output"):
        os.makedirs(f"{inference_path}/inference_output")
    if cfg['log_embedds']:
        embedds_pd = log_embedds(cfg,net,all_data['infere_rna_seq'])
        embedds_pd.to_csv(f"{inference_path}/inference_output/{cfg['model_name']}_embedds.csv")
    prepare_inference_results_tcga(cfg,predicted_labels,logits,all_data,max_len)
    all_data["infere_rna_seq"].to_csv(f"{inference_path}/inference_output/{cfg['model_name']}_inference_results.csv")
    if cfg['tensorboard']:
        writer.close()
    return predicted_labels