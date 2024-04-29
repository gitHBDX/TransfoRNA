
import logging
import math
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.modules.normalization import LayerNorm

logger = logging.getLogger(__name__)

def circulant_mask(n: int, window: int) -> torch.Tensor:
    """Calculate the relative attention mask, calculated once when model instatiated, as a subset of this matrix
    will be used for a input length less than max.
    i,j represent relative token positions in this matrix and in the attention scores matrix,
     this mask enables attention scores to be set to 0 if further than the specified window length

        :param n: a fixed parameter set to be larger than largest max sequence length across batches
        :param window: [window length],
        :return relative attention mask
    """
    circulant_t = torch.zeros(n, n)
    # [0, 1, 2, ..., window, -1, -2, ..., window]
    offsets = [0] + [i for i in range(window + 1)] + [-i for i in range(window + 1)]
    if window >= n:
        return torch.ones(n, n)
    for offset in offsets:
        # size of the 1-tensor depends on the length of the diagonal
        circulant_t.diagonal(offset=offset).copy_(torch.ones(n - abs(offset)))
    return circulant_t


class SelfAttention(nn.Module):

    """normal query, key, value based self attention but with relative attention functionality
    and a learnable bias encoding relative token position which is added to the attention scores before the softmax"""

    def __init__(self, config: DictConfig, relative_attention: int):
        """init self attention weight of each key, query, value and output projection layer.

        :param config: model config
        :type config: ConveRTModelConfig
        """
        super().__init__()

        self.config = config
        self.query = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.key = nn.Linear(config.num_embed_hidden, config.num_attention_project)
        self.value = nn.Linear(config.num_embed_hidden, config.num_attention_project)

        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(
            config.num_attention_project, config.num_embed_hidden
        )
        self.bias = torch.nn.Parameter(torch.randn(config.n), requires_grad=True)
        stdv = 1.0 / math.sqrt(self.bias.data.size(0))
        self.bias.data.uniform_(-stdv, stdv)
        self.relative_attention = relative_attention
        self.n = self.config.n
        self.half_n = self.n // 2
        self.register_buffer(
            "relative_mask",
            circulant_mask(config.tokens_len, self.relative_attention),
        )

    def forward(
        self, attn_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """calculate self-attention of query, key and weighted to value at the end.
        self-attention input is projected by linear layer at the first time.
        applying attention mask for ignore pad index attention weight. Relative attention mask
        applied and a learnable bias added to the attention scores.
        return value after apply output projection layer to value * attention

        :param attn_input: [description]
        :type attn_input: [type]
        :param attention_mask: [description], defaults to None
        :type attention_mask: [type], optional
        :return: [description]
        :rtype: [type]
        """
        self.T = attn_input.size()[1]
        # input is B x max seq len x n_emb
        _query = self.query.forward(attn_input)
        _key = self.key.forward(attn_input)
        _value = self.value.forward(attn_input)

        # scaled dot product
        attention_scores = torch.matmul(_query, _key.transpose(1, 2))
        attention_scores = attention_scores / math.sqrt(
            self.config.num_attention_project
        )

        # Relative attention

        # extended_attention_mask = attention_mask.to(attention_scores.device)  # fp16 compatibility
        extended_attention_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        attention_scores = attention_scores + extended_attention_mask

        # fix circulant_matrix to matrix of size 60 x60 (max token truncation_length,
        # register as buffer, so not keep creating masks of different sizes.

        attention_scores = attention_scores.masked_fill(
            self.relative_mask.unsqueeze(0)[:, : self.T, : self.T] == 0, float("-inf")
        )

        # Learnable bias vector is used of max size,for each i, different subsets of it are added to the scores, where the permutations
        # depend on the relative position (i-j). this way cleverly allows no loops. bias vector is 2*max truncation length+1
        # so has a learnable parameter for each eg. (i-j) /in {-60,...60} .

        ii, jj = torch.meshgrid(torch.arange(self.T), torch.arange(self.T))
        B_matrix = self.bias[self.n // 2 - ii + jj]

        attention_scores = attention_scores + B_matrix.unsqueeze(0)

        attention_scores = self.softmax(attention_scores)
        output = torch.matmul(attention_scores, _value)

        output = self.output_projection(output)

        return [output,attention_scores]  # B x T x num embed hidden 



class FeedForward1(nn.Module):
    def __init__(
        self, input_hidden: int, intermediate_hidden: int, dropout_rate: float = 0.0
    ):
        #          512         2048

        super().__init__()

        self.linear_1 = nn.Linear(input_hidden, intermediate_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(intermediate_hidden, input_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.gelu(self.linear_1(x))
        return self.linear_2(self.dropout(x))


class SharedInnerBlock(nn.Module):
    def __init__(self, config: DictConfig, relative_attn: int,exp_flag:bool = False):
        super().__init__()

        self.config = config
        self.self_attention = SelfAttention(config, relative_attn)
        self.norm1 = LayerNorm(config.num_embed_hidden)  # 512
        self.dropout = nn.Dropout(config.dropout)
        self.ff1 = FeedForward1(
            config.num_embed_hidden, config.feed_forward1_hidden, config.dropout
        )
        self.norm2 = LayerNorm(config.num_embed_hidden)

    def forward(self, x: torch.Tensor, attention_mask: int) -> torch.Tensor:

        new_values_x,attn_scores = self.self_attention(x, attention_mask=attention_mask)
        x = x+new_values_x
        x = self.norm1(x)
        x = x + self.ff1(x)
        return self.norm2(x),attn_scores


# pretty basic, just single head. but done many times, stack to have another dimension (4 with batches).# so get stacks of B x H of attention scores T x T..
# then matrix multiply these extra stacks with the v
# (B xnh)x T xT . (Bx nh xTx hs) gives (B Nh) T x hs stacks. now  hs is set to be final dimension/ number of heads, so reorder the stacks (concatenating them)
# can have optional extra projection layer, but doing that later


class MultiheadAttention(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_attn_proj = config.num_embed_hidden * config.num_attention_heads
        self.attention_head_size = int(self.num_attn_proj / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.key = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.value = nn.Linear(config.num_embed_hidden, self.num_attn_proj)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(hidden_states)
            .view(B, T, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        attention_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

            attention_scores = attention_scores + attention_mask

        attention_scores = F.softmax(attention_scores, dim=-1)

        attention_scores = self.dropout(attention_scores)

        y = attention_scores @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.num_attn_proj)

        return y


class PositionalEncoding(nn.Module):
    def __init__(self, model_config: DictConfig,):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=model_config.dropout)
        self.num_embed_hidden = model_config.num_embed_hidden
        pe = torch.zeros(model_config.tokens_len, self.num_embed_hidden)
        position = torch.arange(
            0, model_config.tokens_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.num_embed_hidden, 2).float()
            * (-math.log(10000.0) / self.num_embed_hidden)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class RNAFFrwd(
    nn.Module
):  # params are not shared for context and reply. so need two sets of weights
    """Fully-Connected 3-layer Linear Model"""

    def __init__(self, model_config: DictConfig):
        """
        :param input_hidden: first-hidden layer input embed-dim
        :type input_hidden: int
        :param intermediate_hidden: layer-(hidden)-layer middle point weight
        :type intermediate_hidden: int
        :param dropout_rate: dropout rate, defaults to None
        :type dropout_rate: float, optional
        """
        # paper specifies,skip connections,layer normalization, and orthogonal initialization

        super().__init__()
        # 3,679,744 x2 params
        self.rna_ffwd_input_dim = (
            model_config.num_embed_hidden * model_config.num_attention_heads
        )
        self.linear_1 = nn.Linear(self.rna_ffwd_input_dim, self.rna_ffwd_input_dim)
        self.linear_2 = nn.Linear(self.rna_ffwd_input_dim, self.rna_ffwd_input_dim)

        self.norm1 = LayerNorm(self.rna_ffwd_input_dim)
        self.norm2 = LayerNorm(self.rna_ffwd_input_dim)
        self.final = nn.Linear(self.rna_ffwd_input_dim, model_config.num_embed_hidden)
        self.orthogonal_initialization()  # torch implementation works perfectly out the box,

    def orthogonal_initialization(self):
        for l in [
            self.linear_1,
            self.linear_2,
        ]: 
            torch.nn.init.orthogonal_(l.weight)

    def forward(self, x: torch.Tensor, attn_msk: torch.Tensor) -> torch.Tensor:
        sentence_lengths = attn_msk.sum(1)

        # adding square root reduction projection separately as not a shared.
        # part of the diagram torch.Size([Batch, scent_len, embedd_dim])

        # x has dims B x T x 2*d_emb
        norms = 1 / torch.sqrt(sentence_lengths.double()).float()  # 64
        # TODO: Aggregation is done on all words including the masked ones
        x = norms.unsqueeze(1) * torch.sum(x, dim=1)  # 64 x1024

        x = x + F.gelu(self.linear_1(self.norm1(x)))
        x = x + F.gelu(self.linear_2(self.norm2(x)))

        return F.normalize(self.final(x), dim=1, p=2)  # 64 512


class RNATransformer(nn.Module):
    def __init__(self, model_config: DictConfig,exp_flag:bool=False):
        super().__init__()
        self.num_embedd_hidden = model_config.num_embed_hidden
        self.encoder = nn.Embedding(
            model_config.vocab_size, model_config.num_embed_hidden
        )
        self.model_input = model_config.model_input
        if 'baseline' not in self.model_input:
            # positional encoder
            self.pos_encoder = PositionalEncoding(model_config)

            self.transformer_layers = nn.ModuleList(
                [
                    SharedInnerBlock(model_config, int(window/model_config.window),exp_flag=exp_flag)
                    for window in model_config.relative_attns[
                        : model_config.num_encoder_layers
                    ]
                ]
            )
            self.MHA = MultiheadAttention(model_config)
            # self.concatenate = FeedForward2(model_config)

            self.rna_ffrwd = RNAFFrwd(model_config)
            self.pad_id = 0

    def forward(self, x:torch.Tensor,exp_flag: bool=False) -> torch.Tensor:
        if x.is_cuda:
            long_tensor = torch.cuda.LongTensor
        else:
            long_tensor = torch.LongTensor
            
        if not exp_flag:
            embedds = self.encoder(x)
        if 'baseline' not in self.model_input:
            if not exp_flag:
                output = self.pos_encoder(embedds)
                attention_mask = (x != self.pad_id).int()
            else:
                exp_profile_length = int(x.shape[1]/self.num_embedd_hidden)
                output = x.reshape((x.shape[0],exp_profile_length,self.num_embedd_hidden))
                attention_mask = torch.ones(size=(output.shape[0],output.shape[1])).type(long_tensor)

            for l in self.transformer_layers:
                output,attn_scores = l(output, attention_mask)
            output = self.MHA(output)
            output = self.rna_ffrwd(output, attention_mask)
            return output,attn_scores
        else:
            embedds = torch.flatten(embedds,start_dim=1)
            return embedds,None
            
class GeneExpFeedForward(nn.Module):
    def __init__(self, model_config: Dict):
        super(GeneExpFeedForward, self).__init__()
        self.ff_input_dim = model_config.ff_input_dim
        self.ff_hidden_dim = model_config.ff_hidden_dim
        current_dim = self.ff_input_dim
        self.layers = nn.ModuleList()
        for hdim in self.ff_hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.LayerNorm(hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, model_config.num_embed_hidden))
        self.orthogonal_initialization()

    def orthogonal_initialization(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, nn.Linear):
                out = F.relu(out)

        return F.normalize(out, dim=1, p=2)

class GeneEmbeddModel(nn.Module):
    def __init__(
        self, main_config: DictConfig,
    ):
        super().__init__()
        self.train_config = main_config["train_config"]
        self.model_config = main_config["model_config"]
        self.device = self.train_config.device
        self.model_input = self.model_config["model_input"]
        self.false_input_perc = self.model_config["false_input_perc"]
        #adjust n (used to add rel bias on attn scores)
        self.model_config.n = self.model_config.tokens_len*2+1
        self.transformer_layers = RNATransformer(self.model_config)
        #save tokens_len of sequences to be used to split ids between transformers
        self.tokens_len = self.model_config.tokens_len
        #reassign tokens_len and vocab_size to init a new transformer
        #more clean solution -> RNATransformer and its children should 
        # have a flag input indicating which transformer
        self.model_config.tokens_len = self.model_config.second_input_token_len
        self.model_config.n = self.model_config.tokens_len*2+1
        self.seq_vocab_size = self.model_config.vocab_size
        #this differs between both models not the token_len/ss_token_len
        self.model_config.vocab_size = self.model_config.second_input_vocab_size 

        if 'exp' not in self.model_input:
            self.second_input_model = RNATransformer(self.model_config)
        else:
            self.second_input_model = RNATransformer(self.model_config,exp_flag=True)

        #num_transformers refers to using either one model or two in parallel
        self.num_transformers = 2
        if self.model_input == 'seq':
            self.num_transformers = 1
        # could be moved to model
        self.weight_decay = self.train_config.l2_weight_decay
        if 'baseline' in self.model_input:
            self.num_transformers = 1
            num_nodes = self.model_config.num_embed_hidden*self.tokens_len
            self.final_clf_1 = nn.Linear(num_nodes,self.model_config.num_classes)
        else:
            #setting classification layer
            num_nodes = self.num_transformers*self.model_config.num_embed_hidden
            if self.num_transformers == 1:
                self.final_clf_1 = nn.Linear(num_nodes,self.model_config.num_classes)
            else:
                self.final_clf_1 = nn.Linear(num_nodes,num_nodes)
                self.final_clf_2 = nn.Linear(num_nodes,self.model_config.num_classes)
                self.relu = nn.ReLU()
                self.BN = nn.BatchNorm1d(num_nodes)
                self.dropout = nn.Dropout(0.6)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def distort_input(self,x):
        for sample_idx in range(x.shape[0]):
            seq_length = x[sample_idx,-1]
            num_tokens_flipped = int(self.false_input_perc*seq_length)
            max_start_flip_idx = seq_length - num_tokens_flipped

            random_feat_idx = random.randint(0,max_start_flip_idx-1)
            x[sample_idx,random_feat_idx:random_feat_idx+num_tokens_flipped] = \
                    torch.tensor(np.random.choice(range(1,self.seq_vocab_size-1),size=num_tokens_flipped,replace=True))

            x[sample_idx,random_feat_idx+self.tokens_len:random_feat_idx+self.tokens_len+num_tokens_flipped] = \
                    torch.tensor(np.random.choice(range(1,self.model_config.second_input_vocab_size-1),size=num_tokens_flipped,replace=True))
        return x
        
    def forward(self, x,train=False):
        if self.device == 'cuda':
            long_tensor = torch.cuda.LongTensor
            float_tensor = torch.cuda.FloatTensor
        else:
            long_tensor = torch.LongTensor
            float_tensor = torch.FloatTensor
        if train:
            if self.false_input_perc > 0:
                x = self.distort_input(x)

        gene_embedd,attn_scores_first = self.transformer_layers(
            x[:, : self.tokens_len].type(long_tensor)
        )
        attn_scores_second = None
        if 'exp' not in self.model_input:
            second_input_embedd,attn_scores_second = self.second_input_model(
                    x[:, self.tokens_len :-1].type(long_tensor)
                )
        else:
            second_input_embedd,attn_scores_second = self.second_input_model(
                    x[:, self.tokens_len :-2].type(float_tensor)
                    ,exp_flag=True
                )
        #for tcga: if seq or baseline
        if self.num_transformers == 1:
            activations = self.final_clf_1(gene_embedd)
        else:
            out_clf_1 = self.final_clf_1(torch.cat((gene_embedd, second_input_embedd), 1))
            out = self.BN(out_clf_1)
            out = self.relu(out)
            out = self.dropout(out)
            activations = self.final_clf_2(out)
        
        #create dummy attn scores for baseline
        if 'baseline' in self.model_input:
            attn_scores_first = torch.ones((1,2,2),device=x.device)

        return [gene_embedd, second_input_embedd, activations,attn_scores_first,attn_scores_second]
