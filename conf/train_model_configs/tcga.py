import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

dirname, _ = os.path.split(os.path.dirname(__file__))


@dataclass
class GeneEmbeddModelConfig:

    model_input: str =  "" #will be infered
    
    num_embed_hidden: int = 100 #30 for exp, 100 for rest
    ff_input_dim:int = 0 #is infered later, equals gene expression len
    ff_hidden_dim: List = field(default_factory=lambda: [300]) #300 for exp hico
    feed_forward1_hidden: int = 256
    num_attention_project: int = 64
    num_encoder_layers: int = 1
    dropout: float = 0.2
    n: int = 121
    relative_attns: List = field(default_factory=lambda: [29, 4, 6, 8, 10, 11])
    num_attention_heads: int = 5

    window: int = 2
    # 200 is max rna length.
    # TODO: if tokenizer is overlap, then max_length should be 60
    # otherwise, will get cuda error, maybe dask can help
    max_length: int = 40
    tokens_len: int = math.ceil(max_length / window)
    second_input_token_len: int = 0 # is infered during runtime
    vocab_size: int = 0  # is infered during runtime
    second_input_vocab_size: int = 0  # is infered during runtime
    tokenizer: str = (
        "overlap"  # either overlap or no_overlap or overlap_multi_window
    )

    clf_target:str = 'major_class_hico' # sub_class, major_class, sub_class_hico or major_class_hico. hico = high confidence
    num_classes: int = 0 #will be infered during runtime
    class_mappings:List = field(default_factory=lambda: [])#will be infered during runtime
    class_weights :List = field(default_factory=lambda: [])
    # how many extra window sizes other than deafault window
    temperatures: List = field(default_factory=lambda: [0,10])
    
    tokens_mapping_dict: Dict = None
    false_input_perc:float = 0.0


@dataclass
class GeneEmbeddTrainConfig:
    dataset_path_train: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/TCGA__ngs__miRNA_log2RPM-24.04.0__var.csv'
    precursor_file_path: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA/data/HBDxBase.csv'
    mapping_dict_path: str = '/media/ftp_share/hbdx/data_for_upload/TransfoRNA//data/subclass_to_annotation.json'
    device: str = "cuda"
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

    max_epochs: int  = 2000

    
