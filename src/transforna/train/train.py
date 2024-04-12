from typing import Dict

from omegaconf import DictConfig

from ..callbacks.metrics import accuracy_score
from ..dataset.seq_tokenizer import SeqTokenizer
from ..score.score import (compute_score_benchmark, compute_score_tcga,
                           infere_additional_test_data)
from ..utils.utils import *
from ..utils.file import load, save


def compute_cv(cfg:DictConfig,path:str):

    summary_pd = pd.DataFrame(index=np.arange(cfg["num_replicates"]),columns = ['B. Acc','Dur'])
    for seed_no in range(cfg["num_replicates"]):
        print(f"Currently training replicate {seed_no}")
        cfg["seed"] = seed_no
        #only log embedds of the last replicate
        if seed_no == cfg["num_replicates"] - 1:
            cfg["log_embedds"] = True
        else:
            cfg["log_embedds"] = False
        test_score,net = train(cfg,path=path)                
        convrg_epoch = np.where(net.history[:,'val_acc_best'])[0][-1]
        convrg_dur = sum(net.history[:,'dur'][:convrg_epoch+1])
        summary_pd.iloc[seed_no] = [test_score,convrg_dur]
    
    save(path=path+'/summary_pd',data=summary_pd)
    
    return

def train(cfg:Dict= None,path:str = None):
    if cfg['tensorboard']:
        from ..tbWriter import writer
    #set seed
    set_seed_and_device(cfg["seed"],cfg["device_number"])

    ad = load(cfg["train_config"].dataset_path_train)

    #instantiate dataset class
    
    if cfg["task"] in ["premirna","sncrna"]:
        dataset_class = SeqTokenizer(ad.var,cfg)
        test_data = load(cfg["train_config"].dataset_path_test)
        #prepare data for training and inference
        all_data = prepare_data_benchmark(dataset_class,test_data,cfg)
    else: 
        ad = append_model_input(ad,cfg)
        dataset_class = SeqTokenizer(ad.var,cfg)
        all_data = prepare_data_tcga(dataset_class,cfg)

    #sync skorch config with params in train and model config
    sync_skorch_with_config(cfg["model"]["skorch_model"],cfg)

     # instantiate skorch model
    net = instantiate_predictor(cfg["model"]["skorch_model"], cfg,path)
    
    #train
    #if train_split is none, then discard valid_ds
    net.fit(all_data["train_data"],all_data["train_labels_numeric"],all_data["valid_ds"])

    #compute scores and log embedds
    if cfg['task'] == 'tcga':
        test_score = compute_score_tcga(net, all_data,path,cfg)
    else:
        test_score = compute_score_benchmark(net, path,all_data,accuracy_score,cfg)
    #only for premirna
    if "additional_testset" in all_data:
        infere_additional_test_data(net,all_data["additional_testset"])

    #log train and model HP to curr run dir    
    save(data=OmegaConf.to_container(cfg, resolve=True),path=path+'/meta/hp_settings')

    if cfg['tensorboard']:
        writer.close()
    return test_score,net
