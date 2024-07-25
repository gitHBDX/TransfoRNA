#%%
import pandas as pd

models = ["seq", "seq-seq", "seq-struct", "seq-reverse"] 
model_names = ["Seq", "Seq-Seq", "Seq-Struct", "Seq-Reverse"] 

# NOTE: refer to work instruction "How to train TransfoRNA.docx" for how the models were trained and the inference results were generated
results_folder = "/media/ftp_share/hbdx/analysis/transforna_LC/TransfoRNA_FULL/sub_class/inference/2024-06-28__240627_slovakia_all_tubes" 


#%%
# load and concatenate inference results

df = None
for i in range(len(models)):
    print(models[i])
    df_ = pd.read_csv(f"{results_folder}/{model_names[i]}/inference_output/{models[i]}_inference_results.csv")
    df_["Model"] = models[i]
    df_.rename(columns={"Sequences, Max Length=40": "Sequence"}, inplace=True)
    df_.rename(columns={"Sequences, Max Length=41": "Sequence"}, inplace=True)
    df = pd.concat([df, df_], axis=0)
df

#%%
# subset to seq model output
df_seq = df[df["Model"] == "seq"]
df_seq


#%%
# check duplicated Original Sequence
df_seq[df_seq.duplicated(subset=["Original_Sequence"], keep='first')]

df_seq[df_seq['Original_Sequence'] == 'GTGAAAAAGAACCTGAAACCGTGTACGTACAAGCA']



#%%
