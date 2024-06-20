import logging
from typing import Dict

from anndata import AnnData
from omegaconf import DictConfig, OmegaConf

from ..callbacks.metrics import accuracy_score
from ..novelty_prediction.id_vs_ood_entropy_clf import compute_entropies
from ..novelty_prediction.id_vs_ood_nld_clf import compute_nlds
from ..processing.augmentation import DataAugmenter
from ..processing.seq_tokenizer import SeqTokenizer
from ..processing.splitter import *
from ..processing.splitter import DataSplitter
from ..score.score import (compute_score_benchmark, compute_score_tcga,
                           infere_additional_test_data)
from ..utils.file import load, save
from ..utils.utils import (instantiate_predictor, prepare_data_benchmark,
                           set_seed_and_device, sync_skorch_with_config)

logger = logging.getLogger(__name__)

def compute_cv(cfg:DictConfig,path:str,output_dir:str):

    summary_pd = pd.DataFrame(index=np.arange(cfg["num_replicates"]),columns = ['B. Acc','Dur'])
    for seed_no in range(cfg["num_replicates"]):
        print(f"Currently training replicate {seed_no}")
        cfg["seed"] = seed_no
        test_score,net = train(cfg,path=path,output_dir=output_dir)                
        convrg_epoch = np.where(net.history[:,'val_acc_best'])[0][-1]
        convrg_dur = sum(net.history[:,'dur'][:convrg_epoch+1])
        summary_pd.iloc[seed_no] = [test_score,convrg_dur]
    
    save(path=path+'/summary_pd',data=summary_pd)
    
    return

def train(cfg:Dict= None,path:str = None,output_dir:str = None):
    if cfg['tensorboard']:
        from ..callbacks.tbWriter import writer
    #set seed
    set_seed_and_device(cfg["seed"],cfg["device_number"])

    dataset = load(cfg["train_config"].dataset_path_train)
    
    if isinstance(dataset,AnnData):
        dataset = dataset.var
    else:
        dataset.set_index('sequence',inplace=True)

    #instantiate dataset class
    
    if cfg["task"] in ["premirna","sncrna"]:
        tokenizer = SeqTokenizer(dataset,cfg)
        test_ad = load(cfg["train_config"].dataset_path_test)
        #prepare data for training and inference
        all_data = prepare_data_benchmark(tokenizer,test_ad,cfg)
    else: 
        df = DataAugmenter(dataset,cfg).get_augmented_df()
        tokenizer = SeqTokenizer(df,cfg)
        all_data = DataSplitter(tokenizer,cfg).prepare_data_tcga()

    #sync skorch config with params in train and model config
    sync_skorch_with_config(cfg["model"]["skorch_model"],cfg)

     # instantiate skorch model
    net = instantiate_predictor(cfg["model"]["skorch_model"], cfg,path)
    
    #train
    #if train_split is none, then discard valid_ds
    net.fit(all_data["train_data"],all_data["train_labels_numeric"],all_data["valid_ds"])
    
    #log train and model HP to curr run dir    
    save(data=OmegaConf.to_container(cfg, resolve=True),path=path+'/meta/hp_settings.yaml')

    #compute scores and log embedds
    if cfg['task'] == 'tcga':
        test_score = compute_score_tcga(net, all_data,path,cfg)
        compute_nlds(output_dir)
        compute_entropies(output_dir)
    else:
        test_score = compute_score_benchmark(net, path,all_data,accuracy_score,cfg)
    #only for premirna
    if "additional_testset" in all_data:
        infere_additional_test_data(net,all_data["additional_testset"])



    if cfg['tensorboard']:
        writer.close()
    return test_score,net
