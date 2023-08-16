from dataclasses import dataclass, field
from typing import List

@dataclass
class GeneEmbeddModelConfig:
    # input dim for the embedding and positional encoders
    # as well as all k,q,v input/output dims
    num_embed_hidden: int = 256
    ff_hidden_dim: List = field(default_factory=lambda: [1200, 800])
    feed_forward1_hidden: int = 1024
    num_attention_project: int = 64
    num_encoder_layers: int = 1
    dropout: float = 0.5
    n: int = 121
    relative_attns: List = field(default_factory=lambda: [int(112), int(112), 6*3, 8*3, 10*3, 11*3])
    num_attention_heads: int = 1

    window: int = 2
    # 200 is max rna length.
    # TODO: if tokenizer is overlap, then max_length should be 60
    # otherwise, will get cuda error, maybe dask can help
    max_length: int = 0 #will be infered later
    tokens_len: int = 0 #will be infered later
    second_input_token_len: int = 0 # is infered in runtime
    vocab_size: int = 0  # is infered in runtime
    second_input_vocab_size: int = 0  # is infered in runtime
    tokenizer: str = (
        "overlap"  # either overlap or no_overlap or overlap_multi_window
    )
    num_classes: int = 0 #will be infered in runtime
    class_weights :List = field(default_factory=lambda: [])
    tokens_mapping_dict: dict = None

    #false input percentage
    false_input_perc:float = 0.1
    model_input: str = "seq-struct"

@dataclass
class GeneEmbeddTrainConfig:
    dataset_path_train: str = "/data/hbdx_ldap_local/analysis/data/premirna/train"
    dataset_path_test: str = "/data/hbdx_ldap_local/analysis/data/premirna/test"
    datset_path_additional_testset: str = "/data/hbdx_ldap_local/analysis/data/premirna/"
    labels_mapping_path:str = "/data/hbdx_ldap_local/analysis/data/premirna/labels_mapping_dict.pkl"
    device: str = "cuda"
    l2_weight_decay: float = 1e-5
    batch_size: int = 64

    batch_per_epoch: int = 0 #will be infered later
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
    train_epoch: int = 3000
    max_epochs: int  = 3500

    freeze_flag:bool = False

