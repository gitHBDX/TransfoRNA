# TransfoRNA
TransfoRNA is a **Bioinformatics** and **machine learning** tool based on **Transformers** to provide annotations for 11 major classes (miRNA, rRNA, tRNA, snoRNA, protein 
-coding/mRNA, lncRNA, and YRNA, piRNA,snRNA,snoRNA and vtRNA) and 1225 sub-classes 
for **small RNAs and RNA fragments**. These are typically found in RNA-seq NGS (next generation sequencing) data.

TransfoRNA can be trained on just the RNA sequences and optionally on additional information such as secondary structure and expression. The result is a major and sub-class assignment combined with a novelty score (Normalized Levenestein Distance) that quantifies the difference between the query sequence and the closest match found in the training set. Based on that it deceids if the query sequene is novel or familiar. TransfoRNA uses a small curated set of ground truth labels obtained from common knowledge-based Bioinformatics tools, including the mapping to databases and the human genome. 

 
## Dataset (Objective):
- **The Cancer Genome Atlas, [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)** offers sequencing data of small RNAs and is used to evaluate TransfoRNAs performance (classification of 252 sub-classes belonging to 7 major classes).
  - Sequences are annotated based on a knowledge-based annotation approach that provides annotations for 1k+ different sub-classes.
  - Knowledge-based annotations are divided into three sets of varying confidence levels: a **high-confidence (HICO)** set, a **low-confidence (LOCO)** set, and a **non-annotated (NA)** set for sequences that could not be annotated at all. Only HICO annotations are used for training.
  - HICO RNAs cover 780 sub-classes and constitute ~9% of all RNAs found in TCGA. LOCO and NA sets comprise 60% and 31% of RNAs, respectively.
  - HICO RNAs are further divided into **in-distribution, ID** (252 sub-classes) and **out-of-distribution, OOD** (528 sub-classes) sets.
    - Criteria for ID and OOD:  Sub-class containing more than 10 sequences are considered ID, otherwise OOD.
  - An additional **artificial affix set, AA** contains 788 sequences known to be technical artefacts.
    
## Models
There are 5 models currently available, each with different input encoders.
 - Baseline: 
    - Input: (single input) Sequence
    - Model: An embedding layer that converts sequences into vectors followed by a classification feed forward layer.
 - Seq: 
    - Input: (single input) Sequence
    - Model: A transformer based encoder model.
 - Seq-Seq:
    - Input: (dual inputs) Sequence divided into even and odd tokens.
    - Model: A transformer encoder is placed for odd tokens and another for even tokens.
 - Seq-Struct:
    - Input: (dual inputs) Sequence + Secondary structure
    - Model: A transformer encoder for the sequence and another for the secondary structure.
 - Seq-Rev (best performant):
    - Input: (dual inputs) Sequence
    - Model: A transformer encoder for the sequence and another for the sequence reversed.
    
<img width="948" alt="Screenshot 2023-08-16 at 16 39 20" src="https://github.com/gitHBDX/TransfoRNA-Framework/assets/82571392/d7d092d8-8cbd-492a-9ccc-994ffdd5aa5f">


    
## Repo Structure
- configs: Contains the configurations of each model, training and inference settings.
 
  The `configs/main_config.yaml` file offers options to change the task, the training settings and the logging. The following shows all the options and permitted values for each option.

   <img width="500" alt="Screenshot 2022-12-16 at 15 56 41" src="https://user-images.githubusercontent.com/82571392/208125570-30b5719c-cb6d-4e39-bbb6-02611336cd6a.png">

- [tcga scripts](https://github.com/gitHBDX/TransfoRNA/blob/master/tcga_scripts/readme.md) are scripts that offer various detailed analyses on TCGA.

- transforna
    - Contains the transforna package which includes data preprocessing, splitting, model training and results logging. 
    - An abstract [schematic](https://github.com/gitHBDX/TransfoRNA/blob/master/transforna/readme.md) shows how TransfoRNA modules communicate during a training run.

## Installation

 The `environment.yml` includes all the required packages for TransfoRNA installation. Edit the `prefix` key to point to the conda folder, then run:
 
 ```
 conda env create -f environment.yml
 
 conda activate transforna
 ```
 
 This will create and activate a new conda environment with the name: `transforna`
 
## Inference
For inference, two paths in `configs/inference_settings/default.yaml` have to be edited:
  - `sequences_path`: The full path to a csv file containing the sequences for which annotations are to be inferred.
  - `model_path`: The full path of the model.
  
Also in the `main_config.yaml`, make sure to edit the `model_name` to match the input expected by the loaded model.
  - `model_name`: add the name of the model. One of `"seq"`,`"seq-seq"`,`"seq-struct"`,`"baseline"`, `seq-exp` or `"seq-reverse"` (see above)


Then, run the following command:

```
python main.py inference=True
```

## Train
TransfoRNA requires the input data to be in the form of an Anndata, `ad`, where `ad.var` contains all the sequences and `ad.X` contains the expression (for Seq-Exp Model). The path of the anndata should be appended to the `dataset_path_train` key in `configs/train_model_configs/tcga`.

For training TransfoRNA from the root directory: 
```
python main.py
```
Using [Hydra](https://hydra.cc/), any option in the main config can be changed. For instance, to train a Seq-Struct TransfoRNA model without using a validation split:
```
python main.py train_split=False model_name='seq-struct'
```
After training, an output folder is automatically created in the root directory where training is logged. 
The structure of the output folder is chosen by hydra to be `/day/time/results folders`. Results folders are a set of folders created during training:
- `ckpt`: (containing the latest checkpoint of the model)
- `embedds`:
  - Contains a file per each split (train/valid/test/ood/na).
  - Each file is a `csv` containing the sequences plus their embeddings (obtained by the model) as well as the logits. The logits are values the models produce for each sequence, reflecting its confidence of a sequence belonging to a certain class.
- `meta`: A folder containing a `yaml` file with all the hyperparameters used for the current run. 


## Additional Datasets (Objective):
- sncRNA, collected from [RFam](https://rfam.org/) (classification of RNA precursors into 13 classes)
- premiRNA [human miRNAs](http://www.mirbase.org)(classification of true vs pseudo precursors)
  
