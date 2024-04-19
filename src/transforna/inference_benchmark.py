
from pathlib import Path
from typing import Dict

from .callbacks.metrics import accuracy_score
from .processing.seq_tokenizer import SeqTokenizer
from .score.score import infer_from_model, infer_testset
from .utils.file import load, save
from .utils.utils import *


def infer_benchmark(cfg:Dict= None,path:str = None):
    if cfg['tensorboard']:
        from .tbWriter import writer

    model = cfg["model_name"]+'_'+cfg['task']

    #set seed
    set_seed_and_device(cfg["seed"],cfg["device_number"])
    #get data
    ad = load(cfg["train_config"].dataset_path_train)

    #instantiate dataset class
    dataset_class = SeqTokenizer(ad.var,cfg)
    test_data = load(cfg["train_config"].dataset_path_test)
    #prepare data for training and inference
    all_data = prepare_data_benchmark(dataset_class,test_data,cfg)



    #sync skorch config with params in train and model config
    sync_skorch_with_config(cfg["model"]["skorch_model"],cfg)

     # instantiate skorch model
    net = instantiate_predictor(cfg["model"]["skorch_model"], cfg,path)
    net.initialize()
    net.load_params(f_params=f'{cfg["inference_settings"]["model_path"]}')

    #perform inference on task specific testset
    if cfg["inference_settings"]["infere_original_testset"]:
        infer_testset(net,cfg,all_data,accuracy_score)
    
    #inference on custom data
    predicted_labels,logits,_,_ = infer_from_model(net,all_data["infere_data"])  
    prepare_inference_results_benchmarck(net,cfg,predicted_labels,logits,all_data)
    save(path=Path(__file__).parent.parent.absolute() / f'inference_results_{model}',data=all_data["infere_rna_seq"])
    if cfg['tensorboard']:
        writer.close()