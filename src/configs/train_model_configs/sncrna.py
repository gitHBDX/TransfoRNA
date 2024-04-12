import os
from dataclasses import dataclass, field
from pickletools import int4
from typing import List


@dataclass
class GeneEmbeddModelConfig:
    # input dim for the embedding and positional encoders
    # as well as all k,q,v input/output dims
    model_input: str = "seq-struct"
    num_embed_hidden: int = 256
    ff_hidden_dim: List = field(default_factory=lambda: [1200, 800])
    feed_forward1_hidden: int = 1024
    num_attention_project: int = 64
    num_encoder_layers: int = 2
    dropout: float = 0.3
    n: int = 121
    window:int  = 4
    relative_attns: List = field(default_factory=lambda: [int(360), int(360)])
    num_attention_heads: int = 4

    
    # 200 is max rna length.
    # TODO: if tokenizer is overlap, then max_length should be 60
    # otherwise, will get cuda error, maybe dask can help
    max_length: int = 0 #will be infered later
    tokens_len: int = 0 #will be infered later
    second_input_token_len:int = 0 # is infered in runtime
    vocab_size: int = 0  # is infered in runtime
    second_input_vocab_size: int = 0  # is infered in runtime
    tokenizer: str = (
        "overlap"  # either overlap or no_overlap or overlap_multi_window
    )
    # how many extra window sizes other than deafault window
    num_classes: int = 0 #will be infered in runtime
    class_weights :List = field(default_factory=lambda: [])
    tokens_mapping_dict: dict = None

    #false input percentage
    false_input_perc:float = 0.2
    
    model_input: str = "seq-struct"


@dataclass
class GeneEmbeddTrainConfig:
    dataset_path_train: str = "/data/hbdx_ldap_local/analysis/data/sncRNA/train.h5ad"
    dataset_path_test: str = "/data/hbdx_ldap_local/analysis/data/sncRNA/test.h5ad"
    labels_mapping_path:str = "/data/hbdx_ldap_local/analysis/data/sncRNA/labels_mapping_dict.pkl"
    device: str = "cuda"
    l2_weight_decay: float = 1e-5
    batch_size: int = 64

    batch_per_epoch:int = 0 #will be infered later
    label_smoothing_sim:float = 0.0
    label_smoothing_clf:float = 0.0
    #pretrain using CL for pretrain_epoch
    pretrain_epochs: int = 1

    # learning rate
    learning_rate: float = 1e-3  # final learning rate ie 'lr annealed to'
    lr_warmup_start: float = 0.1  # start of lr before initial linear warmup section
    lr_warmup_end: float = 1  # end of linear warmup section , annealing begin
    # TODO: 122 is the number of train batches per epoch, should be infered and set
    # warmup batch should be in the form epoch*(train batch per epoch)
    warmup_epoch: int = 10  # how many batches linear warm up for
    final_epoch: int = 20  # final batch of training when want learning rate

    top_k: int = int(
        0.05 * batch_size
    )  # if the corresponding rna/GE appears in the top k, the correctly classified
    label_smoothing: float = 0.0
    cross_val: bool = False
    filter_seq_length:bool = True
    train_epoch: int = 800
    max_epochs:int = 1000

    freeze_flag:bool = False