# TransfoRNA 
TransfoRNA is a **bioinformatics** and **machine learning** tool based on **Transformers** to provide annotations for 11 major classes (miRNA, rRNA, tRNA, snoRNA, protein 
-coding/mRNA, lncRNA, YRNA, piRNA, snRNA, snoRNA and vtRNA) and 1923 sub-classes 
for **human small RNAs and RNA fragments**. These are typically detected by RNA-Seq NGS (next generation sequencing) data. 

Embeddings of RNAs collected from two different [datasets](#datasets); TCGA and inhouse sequenced LC (lung cancer) data can be visualized on our hosted portal [here](www.transforna.com)


TransfoRNA can be trained on just the RNA sequences and optionally on additional information such as secondary structure. The result is a major and sub-class assignment combined with a novelty score (Normalized Levenshtein Distance) that quantifies the difference between the query sequence and the closest match found in the training set. Based on that it decides if the query sequence is novel or familiar. TransfoRNA uses a small curated set of ground truth labels obtained from common knowledge-based bioinformatics tools that map the sequences to transcriptome databases and a reference genome. Using TransfoRNA's framewok, the high confidence annotations in the TCGA dataset can be increased by 3 folds.



## Sections: 
- [Visualize RNAs](https://www.transforna.com) External link!
- [Resources](#resources)
- [Datasets](#datasets)
- [Models](#models)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Inference](#inference)
- [Train](#train)
 
 ## Resources
 The data used for training and the trained models can be downloaded from [here](https://www.dropbox.com/scl/fo/hg3vbw3hzbvyuuhu4fjc6/ALrZ6rUe_9qcKqNgN5Lq7Hg?rlkey=bv40dlw2r4n5wu5adbsxklun0&e=1&dl=0).

  The downloaded folder contains three subfolders (should be kept on the same folder level as `transforna`):
  - `data`:  Contains three files:
    - `TCGA__ngs__miRNA_log2RPM-24.06.0.h5ad` anndata with ~75k sequences and `var` columns containing the knowledge based annotations. 
    - `HBDXBase.csv` containing a list of RNA precursors which are then used for data augmentation. 
    - `subclass_to_annotation.json` holds mappings for every sub-class to major-class.
      
  The models folders contains both benchmark and models trained on tcga.
  - `models`: 
    - `benchmark` : contains benchmark models trained on sncRNA and premiRNA data. (See additional datasets at the bottom)
    - `tcga`: All models trained on the TCGA data; `TransfoRNA_ID` (for testing and validation) and `TransfoRNA_FULL` (the production version) containing  higher RNA major and sub-class coverage. Each of the two folders contain all the models trained seperately on major-class and sub-class.
  - `kba_pipeline`: contains mapping reference data required to run the knowledge based pipeline manually


## Datasets:
- TCGA: **The Cancer Genome Atlas, [TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)** offers sequencing data of small RNAs and is used to train and evaluate TransfoRNAs classification performance 
  - Sequences are annotated based on a knowledge-based annotation approach that provides annotations for ~2k different sub-classes belonging to 11 major classes.
  - Knowledge-based annotations are divided into three sets of varying confidence levels: a **high-confidence (HICO)** set, a **low-confidence (LOCO)** set, and a **non-annotated (NA)** set for sequences that could not be annotated at all. Only HICO annotations are used for training.
  - HICO RNAs cover ~2k sub-classes and constitute 19.6% of all RNAs found in TCGA. LOCO and NA sets comprise 66.9% and 13.6% of RNAs, respectively.
  - HICO RNAs are further divided into **in-distribution, ID** (374 sub-classes) and **out-of-distribution, OOD** (1549 sub-classes) sets.
    - Criteria for ID and OOD:  Sub-class containing more than 8 sequences are considered ID, otherwise OOD.
  - An additional **putative 5' adapter affixes set** contains 294 sequences known to be technical artefacts. The 5’-end perfectly matches the last five or more nucleotides of the 5’-adapter sequence, commonly used in small RNA sequencing.
  - The knowledge-based annotation (KBA) pipline including installation guide is located under `kba_pipline`
- LC: NGS sequencing data from a clinical lung cancer (LC) [study](https://www.jto.org/article/S1556-0864(23)00670-6/fulltext)
  - This dataset was only used for validation. 
- Additional Datasets:
    - sncRNA, collected from [RFam](https://rfam.org/) (classification of RNA precursors into 13 classes)
    - premiRNA [human miRNAs](http://www.mirbase.org)(classification of true vs pseudo precursors)
## Models
There are 5 classifier models currently available, each with different input representation. 
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


*Note: These (Transformer) based models show overlapping and distinct capabilities. Consequently, an ensemble model is created to leverage those capabilities.*


<img width="948" alt="Screenshot 2023-08-16 at 16 39 20" src="https://github.com/gitHBDX/TransfoRNA-Framework/assets/82571392/d7d092d8-8cbd-492a-9ccc-994ffdd5aa5f">

## Repo Structure
`conf`: Contains the configurations of each model, training and inference settings.
 
   The `conf/main_config.yaml` file offers options to change the task, the training settings and the logging. The following shows all the options and permitted values for each option.

   <img width="835" alt="Screenshot 2024-05-22 at 10 19 15" src="https://github.com/gitHBDX/TransfoRNA/assets/82571392/225d2c98-ed45-4ca7-9e86-557a73af702d">

`transforna` contains two folders:

   - `src` folder which contains transforna package. View transforna's architecture [here](https://github.com/gitHBDX/TransfoRNA/blob/master/transforna/src/readme.md).
   - `bin` folder contains all scripts necessary for reproducing manuscript figures.

## Installation
Installation could be done from source or using pip:
  - Installation from source
  
     The `install.sh` is a script that creates an transforna environment in which all the required packages for TransfoRNA are installed. Simply navigate to the root directory and run from terminal:
     
     ```
     #make install script executable
     chmod +x install.sh
    
    
     #run script
     ./install.sh
     ```
  - Installation using pip
   
     ```
     #TODO: Change next line to pip install transforna, once package is on pypi
     pip install git+ssh://git@github.com/gitHBDX/transforna.git
     ```

## Inference
Models could be used for inference in one of three ways:
- **TransfoRNA API**

  In `transforna/src/inference/inference_api.py`, all the functionalities of transforna are offered as APIs. There are two functions of interest:
    - `predict_transforna` : Computes for a set of sequences and for a given model, one of various options; the embeddings, logits, explanatory (similar) sequences, attentions masks or umap coordinates. 
    - `predict_transforna_all_models`: Same as `predict_transforna` but computes the desired option for all the models as well as aggregates the output of the ensemble model.
    Both return a pandas dataframe containing the sequence along with the desired computation. 
  
    Check the script at `src/test_inference_api.py` for a basic demo on how to call the either of the APIs. 
  
- **Inference from repo**

  For inference, two paths in `configs/inference_settings/default.yaml` have to be edited:
    - `sequences_path`: The full path to a csv file containing the sequences for which annotations are to be inferred.
    - `model_path`: The full path of the model. (currently this points to the Seq model)
  
  Also in the `main_config.yaml`, make sure to edit the `model_name` to match the input expected by the loaded model.
    - `model_name`: add the name of the model. One of `"seq"`,`"seq-seq"`,`"seq-struct"`,`"baseline"` or `"seq-rev"` (see above)


  Then, navigate the repositories' root directory and run the following command:
  
  ```
  python transforna/__main__.py inference=True
  ```
  
  After inference, an `inference_output` folder will be created under `outputs/` which will include two files. 
   - `(model_name)_embedds.csv`: contains vector embedding per sequence in the inference set- (could be used for downstream tasks).
     *Note: The embedds of each sequence will only be logged if `log_embedds` in the `main_config` is `True`.* 
   - `(model_name)_inference_results.csv`: Contains columns; Net-Label containing predicted label and Is Familiar? boolean column containing the models' novelty predictor output. (True: familiar/ False: Novel)
     *Note: The output will also contain the logits of the model is `log_logits` in the `main_config` is `True`.* 

- **Inference from huggingface**

  TransfoRNA Models are uploaded to huggingface 🤗:
  [Seq](https://huggingface.co/HBDX/Seq-TransfoRNA)
  [Seq-Seq](https://huggingface.co/HBDX/Seq-Seq-TransfoRNA)
  [Seq-Struct](https://huggingface.co/HBDX/Seq-Struct-TransfoRNA)
  [Seq-Rev](https://huggingface.co/HBDX/Seq-Rev-TransfoRNA)
  
  refer to the script: `TransfoRNA/scripts/test_huggingface_transforna_model.ipynb` for an example.

## Train
TransfoRNA can be trained using input data as Anndata, csv or fasta. If the input is anndata, then `anndata.var` should contains all the sequences. Some changes has to be made (follow `configs/train_model_configs/tcga`):
  
  In `configs/train_model_configs/custom`:
  
   - `dataset_path_train` has to point to the input_data which should contain; a `sequence` column, a `small_RNA_class_annotation` coliumn indicating the major class if available (otherwise should be NaN), `five_prime_adapter_filter` specifies whether the sequence is considered a real sequence or an artifact (`True `for Real and `False` for artifact), a `subclass_name` column containing the sub-class name if available (otherwise should be NaN), and a boolean column `hico` indicating whether a sequence is high confidence or not.
   - If sampling from the precursor is required in order to augment the sub-classes, the `precursor_file_path` should include precursors. Follow the scheme of the HBDxBase.csv and have a look at `PrecursorAugmenter` class in `transforna/src/processing/augmentation.py`
   - `mapping_dict_path` should contain the mapping from sub class to major class. i.e: 'miR-141-5p' to 'miRNA'.
   - `clf_target` sets the classification target of the mopdel and should be either `sub_class_hico` for training on targets in `subclass_name` or `major_class_hico` for training on targets in `small_RNA_class_annotation`. For both, only high confidence sequences are selected for training (based on `hico` column).
    
   In `configs/main_config`, some changes should be made:
   - change `task` to `custom` or to whatever name the `custom.py` has been renamed.
   - set the `model_name` as desired.
  
For training TransfoRNA from the root directory: 
```
python transforna/__main__.py
```

Using [Hydra](https://hydra.cc/), any option in the main config can be changed. For instance, to train a `Seq-Struct` TransfoRNA model without using a validation split:
```
python transforna/__main__.py train_split=False model_name='seq-struct'
```
After training, an output folder is automatically created in the root directory where training is logged. 
  The structure of the output folder is chosen by hydra to be `/day/time/results folders`. Results folders are a set of folders created during training:
  - `ckpt`: (containing the latest checkpoint of the model)
  - `embedds`:
    - Contains a file per each split (train/valid/test/ood/na).
    - Each file is a `csv` containing the sequences plus their embeddings (obtained by the model and represent numeric representation of a given RNA sequence) as well as the logits. The logits are values the models produce for each sequence, reflecting its confidence of a sequence belonging to a certain class.
  - `meta`: A folder containing a `yaml` file with all the hyperparameters used for the current run.
  - `analysis`: contains the learned novelty threshold seperating the in-distribution set(Familiar) from the out of distribution set (Novel).
  - `figures`: some figures are saved containing the Normalized Levenstein Distance NLD, distribution per split.




