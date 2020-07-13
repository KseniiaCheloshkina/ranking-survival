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

paper structure

introduction: survival analysis and aplications. how it is usually solved (sometimes with binary classification for specified time or multiclass)

related research:
- evaluation of survival analysis models (concordance, nbill, brier score)
- deep learning for survival analysiis (recent studies applied deep learniing for classical models - cox,..., wtte

Proposed approach:
1. motivation for optimizing concordance + examples of such studies
from concordance to binary data generators (in which manner we could compare different observations)
we could find it usefull to include such observations in one batch for training (contrast observations)
2. following this intuition we could use additional cross-entropy loss to specify ....
3. Moreover, optimizing loss on such pairs of observations leads to idea of using learning-to-rank approaches -> contrastive loss with pairs generated in different way
- could not achive the same quality by implicitly optimizing this loss
- stacking these two approaches into one end-to-end neural network, which firstly optimizes only loglikelihood and then "fine-tune" found input-output dependencies by using contrastive loss lead to results outperformed previous two approaches

Experiments:
We used 2 datasets with different sample size (), event rate and available feature space
for METABRIC dataset did not perform search of most appropriate architecture
for KKBOX we perform this selection using val set to select the best performing neural network architecture

time should be integer (discrete time)
n batches is important


Implementation Notes:
- To generate pairs of examples appropriate for proposed method we use specific batch generators. It should be noted that for small datasets (like METABRIC dataset) it is possible to use all available pairs of comparable examples. But when dataset size is big all possible pairs can not fit into memory and also could take enourmous time to iterate over during training. For this reason 2 types of batch generators are implemented in project repo: the one which uses all available data and second which finds only given number of pairs for each training example.
