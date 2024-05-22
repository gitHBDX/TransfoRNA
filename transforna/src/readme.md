This is the transforna package which contains the following modules:

- `train` is the entry point where data preparation, training and results logging is executed.

- `processing` contains all classes used for data augmentation, tokenization and splitting.

- `model` contains the skorch model `skorchWrapper` that wraps the torch model described in model components

- `callbacks` contains the learning rate scheduler, loss function and the metrics used to evaluate the model.

- `score` compute the balanced accuracy of the classification task -major or sub-class- for each of the splits with known labels(train/valid/test).

- `novelty_prediction` contains two novelty metrics; entropy based(obsolete) and Normalized Levenstein Distance, NLD based (current).

- `inference` contains all inference functionalities. check `transforna/scripts/test_inference_api.py` for how-to-use.

A schematic of the TransfoRNA Architecture:


![TransfoRNA Architecture](https://github.com/gitHBDX/TransfoRNA/assets/82571392/a1bfbb1e-32c9-4faf-96ae-46727c27e321)

Model evauation image [source](https://medium.com/@sachinsoni600517/model-evaluation-techniques-in-machine-learning-47ae9fb0ad33)
