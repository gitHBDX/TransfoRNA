from typing import List, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
import os
import numpy as np
import torch
#TOKENS to IDS
SEQ_TOKEN_TO_IDS = {'AA': 9,'AC': 10,'AG': 11,'AT': 3,'CA': 15,'CC': 13,'CG': 14,'CT': 1,'GA': 7,'GC': 12,'GG': 16,'GT': 5,'TA': 2,'TC': 8,'TG': 4,'TT': 6,'pad': 0}
SEQ_SEQ_TOKENS_TO_IDS = {'AA': 5,'AC': 16,'AG': 8,'AT': 6,'CA': 12,'CC': 11,'CG': 15,'CT': 2,'GA': 4,'GC': 14,'GG': 13,'GT': 9,'TA': 7,'TC': 1,'TG': 3,'TT': 10,'pad': 0}
SEQ_REV_TOKENS_TO_IDS = {'AA': 8,'AC': 4,'AG': 10,'AT': 7,'CA': 13,'CC': 5,'CG': 16,'CT': 1,'GA': 3,'GC': 11,'GG': 15,'GT': 14,'TA': 6,'TC': 12,'TG': 2,'TT': 9,'pad': 0}
SEQ_STRUCT_TOKENS_TO_IDS = {'AA': 5,'AC': 3,'AG': 15,'AT': 12,'CA': 4,'CC': 6,'CG': 9,'CT': 1,'GA': 14,'GC': 10,'GG': 16,'GT': 13,'TA': 2,'TC': 8,'TG': 11,'TT': 7,'pad': 0,'((': 3,'(.': 4,')(': 8,'))': 7,').': 6,'.(': 2,'.)': 5,'..': 1}

class Tokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids"]#, "attention_mask"]
    do_upper_case: bool = True

    def __init__(
        self,
        do_upper_case: bool = True,
        model_max_length: int = 30,
        token_to_ids: Dict[str, int] = None,
        **kwargs,
    ):
        self._token_to_id = token_to_ids
        self._id_to_token = {id: token for token, id in self._token_to_id.items()}

        super().__init__(
            model_max_length=model_max_length,
            **kwargs,
        )
        self.do_upper_case = do_upper_case


    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, None)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(None))  # type: ignore[arg-type]

    def _tokenize(self, rnas: str, **kwargs):
        if self.do_upper_case:
            rnas = rnas.upper()
        return list(rnas)

    def get_vocab(self):
        return self._token_to_id.copy()

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(None))  # type: ignore[arg-type]

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, None)

    def save_vocabulary(self, save_directory: str, filename_prefix: str  = None):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)
    
    @property
    def all_tokens(self) -> List[str]:
        return list(self.get_vocab().keys())

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)
    
class RnaTokenizer(Tokenizer):
   
    model_input_names = ["input_ids"]#, "attention_mask"]

    def __init__(
        self,
        nmers: int = 2,
        replace_U_with_T: bool = True,
        do_upper_case: bool = True,
        model_max_length: int = 30,
        model_name: str = "",

        **kwargs,
    ):
        self.model_name = model_name.lower()
        print(f'Tokenizing sequences for {self.model_name} model...')
        token_to_ids = SEQ_STRUCT_TOKENS_TO_IDS if 'struct' in self.model_name else SEQ_SEQ_TOKENS_TO_IDS if 'seq-seq' in self.model_name else SEQ_REV_TOKENS_TO_IDS if 'rev' in self.model_name else SEQ_TOKEN_TO_IDS
        super().__init__(
            do_upper_case=do_upper_case,
            model_max_length=model_max_length,
            token_to_ids=token_to_ids,
            **kwargs,
        )
        self.replace_U_with_T = replace_U_with_T
        self.nmers = nmers
        

    def chunkstring_overlap(self, string):
        return (
            string[0 + i : self.nmers + i] for i in range(0, len(string) - self.nmers + 1, 1)
        )
    
    def _tokenize(self, rnas: str, **kwargs):
        if self.do_upper_case:
            rnas = rnas.upper()
        if self.replace_U_with_T:
            rnas = rnas.replace("U", "T")

        return list(self.chunkstring_overlap(rnas))
    
    def custom_roll(self,arr, n_shifts_per_row):
        '''
        shifts each row of a numpy array according to n_shifts_per_row
        '''
        from numpy.lib.stride_tricks import as_strided

        m = np.asarray(n_shifts_per_row)
        arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy() #need `copy`
        strd_0, strd_1 = arr_roll.strides
        n = arr.shape[1]
        result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))

        return result[np.arange(arr.shape[0]), (n-m)%n]
    
    def __call__(
        self,
        rnas: str,
        return_tensors: str = "pt",
        padding: bool = "max_length",
        truncation: bool = True,
        **kwargs,
    ) -> Dict[str, List[int]]:
        seq_lens = np.array([len(rna) for rna in rnas])
        
        result =  super().__call__(
            rnas,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        rna_token_ids = np.array(result["input_ids"])
        second_token_ids = np.zeros_like(rna_token_ids)

        if 'struct' in self.model_name:
            from transforna import fold_sequences
            rnas_ss = list(fold_sequences(rnas)['structure_37'].values)
            result =  super().__call__(
                rnas_ss,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
            second_token_ids = np.array(result["input_ids"])
        elif 'rev' in self.model_name:
            sample_token_ids_rev = rna_token_ids[:,::-1]
            n_zeros = np.count_nonzero(sample_token_ids_rev==0, axis=1)
            second_token_ids = self.custom_roll(sample_token_ids_rev, -n_zeros)
        
        elif 'seq-seq' in self.model_name:
            phase0 = rna_token_ids[:,::2]
            phase1 = rna_token_ids[:,1::2]
            #in case max_length is an odd number phase 0 will be 1 entry larger than phase 1 @ dim=1 
            if phase0.shape!= phase1.shape:
                phase1 = np.concatenate([phase1,np.zeros(phase1.shape[0])[...,np.newaxis]],axis=1)
            rna_token_ids = phase0
            second_token_ids = phase1
        else:
            #seq
            pass
            

        result['input_ids'] = torch.tensor(np.concatenate([rna_token_ids,second_token_ids,seq_lens[...,np.newaxis]],axis=1))

        return result
        