This is the transforna package which contains the skorch model, data preprocessing and sub/major class classification results logging.

- `train` is the entry point where data preparation, training and results logging is executed.

- `utils` contains all functions used for data preprocessing, including preparing the data according to each models input, data splitting and filtering.

- `model` contains the skorch model which wraps the actual model described in model components

- `dataset` is where the sequences/secondary structures are tokenized. if the task is premirna or sncrna then dataset_benchmark.py is called, otherwise dataset_tcga.py.

- `callbacks` contains the learning rate scheduler, loss function and the metrics used to evaluate the model.
- `score` compute the balanced accuracy of the classification task (either major or sub class)

![Screenshot 2022-12-14 at 18 30 43](https://user-images.githubusercontent.com/82571392/207665995-cae86d2d-78a0-498a-9504-2397aa2da344.png)
