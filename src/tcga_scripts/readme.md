## Folder Structure
- tcga scripts contains the novelty prediciton:
- There are two approaches for novelty introduced. NLD and Entropy 
  - `id_vs_ood_entropy_clf.py`: learns a threshold discerning Familiar vs Novel sequences based on entropy.
  - `id_vs_ood_nld_clf.py`: learns a threshold discerning Familiar vs Novel sequences based on normalized levenstein distance.

    
  
## Results Handler
Results Handler is a class that permiates handling a given run. For instance, the specific sequence splits within the run directory could be selectively read, the anndata the model used for training, the configurations used, the creation of the knn graph which the novelty prediction is based on, and the computation of the umaps. 

To use `ResultsHandler`, make sure to edit the `path` in the `main` function. This path should point to the `embedds` folder of a given run. for example, change:

```
path = None
```

to

```
path = /path/to/embedds/
```

results would be generated but not saved. If the results are required to be saved, change 

```
results:Results_Handler = Results_Handler(path=path,splits=splits)
```
to
```
results:Results_Handler = Results_Handler(path=path,splits=splits,save_results=True)
```
This will save the results in the same level as the `embedds` folder of a given run.
