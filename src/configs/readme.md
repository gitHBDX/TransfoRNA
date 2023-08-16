The `inference_settings` has a default yaml containing four keys:
  -`sequences_path`: The full path of the file containing the sequences for which their annotations are to be infered.
  - `model_path`: the full path of the model to be used for inference.
  - `model_name`: A model name indicating the inputs the model expects. One of `seq`,`seq-seq`,`seq-struct`,`seq-reverse` or `baseline`
  - `infere_original_testset`: True/False indicating whether inference should be computed on the original test set. 

`model` contains the skeleton of the model used, the optimizer, loss function and device. All models are built using [skorch](https://skorch.readthedocs.io/en/latest/?badge=latest)

`train_model_configs` contain the hyperparameters for each dataset; tcga, sncrna and premirna:

  - Each file contains the model and the train config.
    
    - Model config: contains the model hyperparameters, sequence tokenization scheme and allows for choosing the model. 
    
    - Train config: contains training settings such as the learning rate hyper parameters as well as `dataset_path_train`.
      - `dataset_path_train`: should point to the dataset [(Anndata)](https://anndata.readthedocs.io/en/latest/) used for training.
      
