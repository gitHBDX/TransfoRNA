# TransfoRNA
TransfoRNA is a **bioinformatics** and **machine learning** tool based on **Transformers** to provide annotations for 11 major classes (miRNA, rRNA, tRNA, snoRNA, protein 
-coding/mRNA, lncRNA, YRNA, piRNA, snRNA, snoRNA and vtRNA) and 1225 sub-classes 
for **human small RNAs and RNA fragments**. These are typically detected by RNA-seq NGS (next generation sequencing) data.

TransfoRNA can be trained on just the RNA sequences and optionally on additional information such as secondary structure. The result is a major and sub-class assignment combined with a novelty score (Normalized Levenshtein Distance) that quantifies the difference between the query sequence and the closest match found in the training set. Based on that it decides if the query sequence is novel or familiar. TransfoRNA uses a small curated set of ground truth labels obtained from common knowledge-based bioinformatics tools that map the sequences to transcriptome databases and a reference genome. 


 
## Dataset (Objective):
- **The Cancer Genome Atlas, [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)** offers sequencing data of small RNAs and is used to evaluate TransfoRNAs performance (classification of 278 sub-classes belonging to 11 major classes).
  - Sequences are annotated based on a knowledge-based annotation approach that provides annotations for 1k+ different sub-classes.
  - Knowledge-based annotations are divided into three sets of varying confidence levels: a **high-confidence (HICO)** set, a **low-confidence (LOCO)** set, and a **non-annotated (NA)** set for sequences that could not be annotated at all. Only HICO annotations are used for training.
  - HICO RNAs cover 1225 sub-classes and constitute ~9% of all RNAs found in TCGA. LOCO and NA sets comprise 60% and 31% of RNAs, respectively.
  - HICO RNAs are further divided into **in-distribution, ID** (278 sub-classes) and **out-of-distribution, OOD** (947 sub-classes) sets.
    - Criteria for ID and OOD:  Sub-class containing more than 10 sequences are considered ID, otherwise OOD.
  - An additional **artificial affix set, AA** contains ~250 sequences known to be technical artefacts.
  - The knowledge-based annotation (KBA) pipline including installation guide is located under `kba_pipline`
    
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

   <img width="566" alt="Screenshot 2023-08-17 at 13 43 15" src="https://github.com/gitHBDX/TransfoRNA-Framework/assets/82571392/bd1cdac5-f1d3-45fb-8543-5c2292a2542f">

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
 ## TransfoRNA API
 In `src/transforna/inference_api.py`, all the functionalities of transforna are offered as APIs. There are two functions of interest:
  - `predict_transforna` : Computes for a set of sequences and for a given model, one of various options; the embeddings, logits, explanatory (similar) sequences, attentions masks or umap coordinates. 
  - `predict_transforna_all_models`: Same as `predict_transforna` but computes the desired option for all the models as well as aggregates the output of the ensemble model.
  Both return a pandas dataframe containing the sequence along with the desired computation. 

  Check the script at `src/tests/test_inference_api.py` for a basic demo on how to call the either of the APIs. 
## Inference
For inference, two paths in `configs/inference_settings/default.yaml` have to be edited:
  - `sequences_path`: The full path to a csv file containing the sequences for which annotations are to be inferred.
  - `model_path`: The full path of the model. (currently this points to the Seq model)
  
Also in the `main_config.yaml`, make sure to edit the `model_name` to match the input expected by the loaded model.
  - `model_name`: add the name of the model. One of `"seq"`,`"seq-seq"`,`"seq-struct"`,`"baseline"` or `"seq-reverse"` (see above)


Then, navigate the repositories' root directory and run the following command:

```
python src/main.py inference=True
```

After inference, an `inference_output` in the `src` folder will be created which will then include two files. 
 - The `inference_results_(model_name).csv` that includes the label of each sequence in the inference set and the models' confidence as to whether the sequence is novel (belongs to a class , the model was not trained on) or familiar. 
 - The embedds of each sequence obtained form the model if `log_embedds` in the `main_config` is `True`. 

## Train on custom data
TransfoRNA requires the input data to be in the form of an Anndata, `ad`, where `ad.var` contains all the sequences. Some changes has to be made (follow `configs/train_model_configs/tcga`):

In `configs/train_model_configs/custom`:
- `dataset_path_train` has to point to the anndata. The anndata has to contain .var dataframe which is sequence indexed. The .var columns should contain `small_RNA_class_annotation`: indicating the major class if available (otherwise should be NaN), `five_prime_adapter_filter`: whether the sequence is considered a real sequence or an artifact (`True `for Real and `False` for artifact), `subclass_name` containing the sub-class name if available (otherwise should be NaN), and a boolean column `hico` indicating whether a sequence is high confidence or non.
- If sampling from the precursor is required in order to augment the sub-classes, the `precursor_file_path` should include precursors. Follow the scheme of the HBDxBase.csv and have a look at `get_precursor_info` in `src/transforna/utils/utils.py`
- `mapping_dict_path` should contain the mapping from sub class to major class. i.e: 'miR-141-5p' to 'miRNA'.
- `clf_target` sets the classification target of the mopdel and should either `sub_class_hico` for training on targets in `subclass_name` or `major_class_hico` for training on targets in `small_RNA_class_annotation`.

In configs/main_config, some changes should be made:
- change `task` to `custom`.
- set the `model_name` as desired.

For training TransfoRNA from the root directory: 
```
python src/main.py
```
Using [Hydra](https://hydra.cc/), any option in the main config can be changed. For instance, to train a `Seq-Struct` TransfoRNA model without using a validation split:
```
python src/main.py train_split=False model_name='seq-struct'
```
After training, an output folder is automatically created in the root directory where training is logged. 
The structure of the output folder is chosen by hydra to be `/day/time/results folders`. Results folders are a set of folders created during training:
- `ckpt`: (containing the latest checkpoint of the model)
- `embedds`:
  - Contains a file per each split (train/valid/test/ood/na).
  - Each file is a `csv` containing the sequences plus their embeddings (obtained by the model and represent numeric representation of a given RNA sequence) as well as the logits. The logits are values the models produce for each sequence, reflecting its confidence of a sequence belonging to a certain class.
- `meta`: A folder containing a `yaml` file with all the hyperparameters used for the current run. 


## Additional Datasets (Objective):
- sncRNA, collected from [RFam](https://rfam.org/) (classification of RNA precursors into 13 classes)
- premiRNA [human miRNAs](http://www.mirbase.org)(classification of true vs pseudo precursors)
  
