import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

dirname, _ = os.path.split(os.path.dirname(__file__))
from transformers import PretrainedConfig

@dataclass
class GeneEmbeddModelConfig(PretrainedConfig):

    def __init__(self, model_input:str="",
                 num_embed_hidden:int=100, ff_input_dim:int=0, 
                 ff_hidden_dim:List=[300], feed_forward1_hidden:int=256, 
                 num_attention_project:int=64, num_encoder_layers:int=1, 
                 dropout:float=0.2, n:int=121, relative_attns:List=[29, 4, 6, 8, 10, 11], 
                 num_attention_heads:int=5, window:int=2, max_length:int=40, 
                 second_input_token_len:int=0, vocab_size:int=0, 
                 second_input_vocab_size:int=0, tokenizer:str="overlap", 
                 clf_target:str='sub_class_hico', num_classes:int=0, 
                 class_mappings:List=[], class_weights:List=[], temperatures:List=[0,10], 
                 tokens_mapping_dict:Dict=None, false_input_perc:float=0.0, **kwargs):
        
        self.model_input: str =  model_input #will be infered
        
        self.num_embed_hidden: int = num_embed_hidden #30 for exp, 100 for rest
        self.ff_input_dim:int = ff_input_dim #is infered later, equals gene expression len
        self.ff_hidden_dim: List = ff_hidden_dim #300 for exp hico
        self.feed_forward1_hidden: int = feed_forward1_hidden
        self.num_attention_project: int = num_attention_project
        self.num_encoder_layers: int = num_encoder_layers
        self.dropout: float = dropout
        self.n: int = n
        self.relative_attns: List = relative_attns
        self.num_attention_heads: int = num_attention_heads

        self.window: int = window
        # 200 is max rna length.
        # TODO: if tokenizer is overlap, then max_length should be 60
        # otherwise, will get cuda error, maybe dask can help
        self.max_length: int = max_length
        self.tokens_len: int = math.ceil(self.max_length/self.window)
        self.second_input_token_len: int = second_input_token_len # is infered during runtime
        self.vocab_size: int = vocab_size  # is infered during runtime
        self.second_input_vocab_size: int = second_input_vocab_size  # is infered during runtime
        self.tokenizer: str = tokenizer # either overlap or no_overlap or overlap_multi_window

        self.clf_target:str = clf_target # sub_class, major_class, sub_class_hico or major_class_hico. hico = high confidence
        self.num_classes: int = num_classes #will be infered during runtime
        self.class_mappings:List = class_mappings#will be infered during runtime
        self.class_weights :List = class_weights
        # how many extra window sizes other than deafault window
        self.temperatures: List = temperatures
        
        self.tokens_mapping_dict: Dict = tokens_mapping_dict
        self.false_input_perc:float = false_input_perc
        super().__init__(**kwargs)


@dataclass
class GeneEmbeddTrainConfig:
    dataset_path_train: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.06.0__var.csv'
    precursor_file_path: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/HBDxBase.csv'
    mapping_dict_path: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/subclass_to_annotation.json'
    device: str = "cpu"
    l2_weight_decay: float = 0.05
    batch_size: int = 512

    batch_per_epoch:int  = 0 # is infered during runtime 

    label_smoothing_sim:float = 0.2
    label_smoothing_clf:float = 0.0
    # learning rate
    learning_rate: float = 1e-3  # final learning rate ie 'lr annealed to'
    lr_warmup_start: float = 0.1  # start of lr before initial linear warmup section
    lr_warmup_end: float = 1  # end of linear warmup section , annealing begin
    # TODO: 122 is the number of train batches per epoch, should be infered and set
    # warmup batch should be during the form epoch*(train batch per epoch)
    warmup_epoch: int = 10  # how many batches linear warm up for
    final_epoch: int = 20  # final batch of training when want learning rate

    top_k: int = 10#int(0.1 * batch_size)  # if the corresponding rna/GE appears during the top k, the correctly classified
    cross_val: bool = False
    labels_mapping_path: str = None
    filter_seq_length:bool = False

    num_augment_exp:int = 20
    shuffle_exp: bool = False

    max_epochs: int  = 2

    
