import argparse
import os
import random
import tensorflow as tf
import numpy as np
import batch_generators
import batch_generators_csr


def main(args):
    # set seed
    s = 2
    random.seed(s)
    np.random.seed(s)
    tf.set_random_seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(s)
    #  'inp_shape': (None, 9), - определить самостоятельно


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on given dataset')
    # specify dataset properties

    # specify main model parameters
    parser.add_argument('--dataset_sampling', required=True, type=bool, choices=[True, False], default=False,
                        help='If False use all available pairs for each example in a dataset.'
                             ' Otherwise for each training example sample only a number of pairs '
                             'specified in argument pairs_per_sample (recommended for big datasets)')
    parser.add_argument('--optimization_strategy', required=True, type=str, choices=['base', 'binary', 'contrastive'],
                        default='base',
                        help='Defines loss function. If "base", log-likelihood is optimized.'
                             'If "binary", binary cross-entropy is added to loss function. '
                             'If "contrastive", contrastive loss is added to loss function.')
    parser.add_argument('--alpha_reg', required=True, type=float, default=1e-6,
                        help='')
    parser.add_argument('--batch_size', required=True, type=int, default=1024,
                        help='')
    # specify additional model attributes
    parser.add_argument('--n_time_bins', required=False, type=int, default=10,
                        help='')
    #  'time_grid': np.linspace(0, 300, 30, dtype=np.int),
    parser.add_argument('--margin_weight', required=False, type=float, default=1e-1,
                        help='')
    # binary weight
    parser.add_argument('--contrastive_weight', required=False, type=float, default=0.5,
                        help='')
    parser.add_argument('--batch_size_contr', required=True, type=int, default=1024,
                        help='if not specified, batch_size is used')
    # specify optimization parameters
    # 'max_lr': 0.002,
    # 'step_size': 13,
    # 'learning_rate_contr_freezed': 0.0001,
    # 'learning_rate_contr': 0.00005,
    # 'momentum': 0.7,
    parser.add_argument('--n_epochs', required=False, type=int, default=30,
                        help='')
    parser.add_argument('--n_epochs_binary', required=False, type=int, default=30,
                        help='if not specified, n_epochs is used')
    parser.add_argument('--n_epochs_contrastive_freezed', required=False, type=int, default=30,
                        help='if not specified, n_epochs is used')
    parser.add_argument('--n_epochs_contrastive', required=False, type=int, default=30,
                        help='if not specified, n_epochs is used')
    args = vars(parser.parse_args())
    print(args)
    main(args)
