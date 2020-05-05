## Survival analysis as a ranking tasks


### Methods

### Experiment setup
To estimate prediction performance of proposed methods we used the next datasets:
- METABRIC
- KKBOX

For METABRIC dataset we used excluded observations with zero duration and next preprocess tha data, using StandardScaler for numerical columns.

Firstly, the datasets were splitted into validation (20% of sample) and training sets stratified by event label and time bin (time bin correspondsto quartile of duration distribution). The training set was further transformed into 5-fold cross-validation dataset. 

Then for each method we selected hyperparameters training models on each training fold of CV dataset and testing the quality on validation set. The hyperparameters were selected ...

Final quality estimates were got when training and evaluating the models using best values of hyperparameters on CV data. 