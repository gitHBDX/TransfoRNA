## Folder Structure
- tcga scripts contains two sub directories:
  - post transforna analysis. These are analysis specific to a given run and offers the following utilites:
    - `art_vs_ood_clf.py`: Artificial Affix, AA vs OOD classification.
    - `before_and_after_transfoRNA.py`: Proportions of HICO/LOCO before and after transforna.
    - `compute_mc_proportion_per_split.py`: Proportions of ID and OOD per major class used for training.
    - `generate_all_tcga_feat_w_predictions.py`: Generating annotations relying on both; knowledge based annotation tool + TransfoRNA.
    - `id_vs_ood_clf.py`: OOD vs ID classification.
    - `save_umaps.py`: plotting UMAPS.
    
  - tcga analysis
    - `5_prime_overlap_aa.py`: Overlap between the 5' adapter and the AA set.
    - `gc_and_len_stats_AA_vs_OOD.py`: Detailed analysis of AA vs OOD samples.
    - `hico_loco_proportion_per_mc.py`: Depition of all major classes used for sub class classification task broken down into proportions of each confidence set.
    - `sequence_alignment_scores_bioython.py`: Computation of alignment scores of major classes and sub-classes.
   
## Post Processing: 
The majority of those scripts mainly rely on the `embedds` folder of a given run. To run any of the scripts, make sure to edit the `path` in the `main` function. This path should point to the `embedds` folder of a given run. for example, change:

```
path = None
```

to

```
path = /path/to/embedds/
```

Depending on the script executed, results would be generated but not saved. If the results are required to be saved, change 

```
results:Results_Handler = Results_Handler(path=path,splits=splits)
```
to
```
results:Results_Handler = Results_Handler(path=path,splits=splits,save_results=True)
```
This will save the results in the same level as the `embedds` folder of a given run.
