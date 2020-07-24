import argparse
import os
import random
import tensorflow as tf
import numpy as np
import json
import pickle
import tqdm
import pandas as pd

import models_hard_mining
from batch_generators_hard_mining import DataGenerator


def main(args):
    if args['verbose'] == 1:
        print("Loading data...")
    # read config
    with open(args['config_path'], 'rb') as f:
        config = json.load(f)
    print("Current config: ")
    print(config)

    # read training data
    with open(args['train_data_path'], 'rb') as f:
        train_data = pickle.load(f)

    # read validation data
    with open(args['val_data_path'], 'rb') as f:
        val_data = pickle.load(f)

    # load custom bottom layers
    eval('from custom_models import ' + args['custom_bottom_function_name'] + 'as custom_bottom')

    # set seed for reproducibility
    if 'seed' in config.keys():
        seed = config['seed']
    else:
        seed = 2
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)

    # check if save_path exists
    if args['save_model'] or args['save_prediction']:
        if not os.path.exists(args['save_path']):
            os.makedirs(args['save_path'])

    # check if all parameters are in config
    check_config(args, train_data, config)

    # initialize model
    if args['verbose'] == 1:
        print("Initialize model...")
    data_gen, model = initialize_model(train_data, config, seed)
    # train model
    if args['verbose'] == 1:
        print("Train model...")
    hist_losses_train, hist_losses_val, pred_train, pred_val = train(
        args, train_data, val_data, config, data_gen, model, seed)
    if args['verbose'] == 1:
        df_losses = pd.DataFrame(data=[hist_losses_train, hist_losses_val], columns=['train_loss', 'val_Loss'])
        # TODO: print(tabulate(df_losses))
    # save prediction
    if args['save_prediction']:
        if args['verbose'] == 1:
            print("Save prediction...")
        with open(args['save_path'] + args["model_type"] + "val_pred.pkl", 'wb') as f:
            pickle.dump(pred_val, f)
        with open(args['save_path'] + args["model_type"] + "train_pred.pkl", 'wb') as f:
            pickle.dump(pred_train, f)


def initialize_model(train_data, config, seed):
    """ Initialize batch generator and model from config """

    inp_shape = (None, train_data['x'].shape[1])
    # initialize batch generator
    dg = DataGenerator(
        x=train_data['x'],
        y=train_data['y'],
        t=train_data['t'],
        n_ex_bin=config['n_ex_bin'],
        n_time_bins=config['n_time_bins'])
    # initialize model
    if args['model_type'] == 'binary':
        model = models_hard_mining.BinaryRankingModel(
            input_shape=inp_shape,
            seed=seed,
            main_network=custom_bottom,
            alpha_reg=config['alpha_reg'],
            alpha_bias_random_mean=config['alpha_bias_random_mean'],
            alpha_random_stddev=config['alpha_random_stddev'],
            beta_random_stddev=config['beta_random_stddev'],
            cross_entropy_weight=config['cross_entropy_weight']
        )
    elif args['model_type'] == 'base':
        model = models_hard_mining.WeibullModel(
            input_shape=inp_shape,
            seed=seed,
            main_network=custom_bottom,
            alpha_reg=config['alpha_reg'],
            alpha_bias_random_mean=config['alpha_bias_random_mean'],
            alpha_random_stddev=config['alpha_random_stddev'],
            beta_random_stddev=config['beta_random_stddev'],
        )
    elif args['model_type'] == 'contrastive':
        model = models_hard_mining.ContrastiveRankingModel(
            input_shape=inp_shape,
            seed=seed,
            main_network=custom_bottom,
            alpha_reg=config['alpha_reg'],
            alpha_bias_random_mean=config['alpha_bias_random_mean'],
            alpha_random_stddev=config['alpha_random_stddev'],
            beta_random_stddev=config['beta_random_stddev'],
            margin_weight=config['margin_weight'],
            contrastive_weight=config['contrastive_weight']
        )
    else:
        raise NotImplementedError("This model type is not implemented. "
                                  "Possible model types: 'base', 'binary', 'contrastive'")
    return dg, model


def initialize_sgd_optimizer(config, model):
    """ Define GradientDescentOptimizer from config """

    global_step = tf.Variable(0, trainable=False)
    # if exponential decay of learning rate is specified in config
    if ('step_rate' not in config.keys()) or (config['step_rate'] == 0) or (
            'decay' not in config.keys()) or (config['decay'] == 1):
        learning_rate = config['learning_rate']
    else:
        learning_rate = tf.train.exponential_decay(
            config['learning_rate'], global_step, config['step_rate'], config['decay'], staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(model.loss)
    return train_op, global_step


def train(args, train_data, val_data, config, data_gen, model, seed):
    # determine optimizer
    train_op, global_step = initialize_sgd_optimizer(config, model)
    # TODO: increment_global_step is not needed???
    increment_global_step = tf.assign(global_step, global_step + 1)
    # lists to store results
    hist_losses_train = []
    hist_losses_val = []
    all_pred_train = []
    all_pred_val = []
    with tf.Session() as sess:
        tf.set_random_seed(seed)
        init = tf.initialize_all_variables()
        sess.run(init)
        # for each epoch
        for i in tqdm.tqdm(range(config['n_epochs'])):
            # initialize generator
            gen = data_gen.get_batch()
            while True:
                try:
                    # get batch data
                    x_batch, y_batch, target = next(gen)
                    feed_dict = {
                        model.x: x_batch,
                        model.t: y_batch[:, 0].reshape((y_batch[:, 0].shape[0], 1)),
                        model.y: y_batch[:, 1].reshape((y_batch[:, 0].shape[0], 1)),
                        model.target: target.reshape((y_batch[:, 0].shape[0], 1)),
                    }
                    # train model on batch
                    _, train_batch_loss = sess.run([train_op, model.loss], feed_dict=feed_dict)
                except StopIteration:
                    # if run out of examples
                    break

            # get prediction for validation data
            pred_val = sess.run(model.o1, feed_dict={model.x: val_data['x']})
            all_pred_val.append(pred_val)
            # get loss for validation data
            val_loss = get_loss_batch(val_data, config, sess, model.loss)
            hist_losses_val.append(val_loss)
            if args['verbose'] == 1:
                print("Val loss at epoch {}: {}".format(i, val_loss))
            # get prediction for training data
            pred_train = sess.run(model.o1, feed_dict={model.x: train_data['x']})
            all_pred_train.append(pred_train)
            # get loss for training data
            train_loss = get_loss_batch(train_data, config, sess, model.loss)
            hist_losses_train.append(train_loss)
        # save model
        if args['verbose'] == 1:
            print("Save model...")
        saver = tf.train.Saver()
        if args['save_model']:
            saver.save(sess, args['save_path'] + args["model_type"] + "_model")
    return hist_losses_train, hist_losses_val, all_pred_train, all_pred_val


def get_loss_batch(data, config, sess, tensor_loss):
    """ Calculate loss on batches """
    dg = DataGenerator(x=data['x'],
                       y=data['y'],
                       t=data['t'],
                       n_ex_bin=config['n_ex_bin'],
                       n_time_bins=config['n_time_bins'])
    gen = dg.get_batch()
    loss = 0
    while True:
        try:
            x_batch, y_batch, target = next(gen)
            loss =+ sess.run(tensor_loss, feed_dict={
                "x:0": x_batch,
                "t:0": y_batch[:, 0].reshape((y_batch[:, 0].shape[0], 1)),
                "y:0": y_batch[:, 1].reshape((y_batch[:, 0].shape[0], 1)),
                "target:0": target.reshape((y_batch[:, 0].shape[0], 1))
            })
        except StopIteration:
            return loss


def check_config(args, train_data, config):
    default_values = {
        "seed": 2,
        "alpha_reg": 1e-6,
        "n_time_bins": 10,
        "n_ex_bin": 30,
        "val_n_ex_bin": 100,
        "n_epochs": 30,
        "step_rate": 10,
        "decay": 0.9,
        "learning_rate": 0.001,
        "alpha_bias_random_mean": 0,
        "alpha_random_stddev": 1,
        "beta_random_stddev": 1
    }
    default_values.update({"time_grid": np.linspace(1, np.max(train_data['t']), 15, dtype=np.int).tolist()})
    if args['model_type'] == "binary":
        default_values.update({"cross_entropy_weight": 1e-1})
    elif args['model_type'] == 'contrastive':
        default_values.update({"margin_weight": 1e-1, "contrastive_weight": 0.5})

    set_keys = config.keys()
    all_keys = default_values.keys()
    missing_keys = set(all_keys) - set(set_keys)
    if missing_keys != set():
        print("Missing config parameters:")
        print(missing_keys)
        print("Use default values for missing config parameters")
        for par in missing_keys:
            config.update({par: default_values[par]})
    print("Final config parameters:")
    print(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on given dataset')
    # running options
    # TODO: type of y
    parser.add_argument('--train_data_path', required=True, type=str,
                        help='''
                        Path to pickle file to train model on. 
                        Format: pickle file contains a dictionary with 3 keys:
                        x - features numpy array (n_samples, n_features)
                        y - event label numpy array (n_samples, )
                        t - duration label numpy array of type float (n_samples, )
                        ''')
    parser.add_argument('--val_data_path', required=False, type=str,
                        help='''
                        Path to pickle file to validate model on
                        Format: pickle file contains a dictionary with 3 keys:
                        x - features numpy array (n_samples, n_features)
                        y - event label numpy array (n_samples, )
                        t - duration label numpy array of type float (n_samples, )
                        ''')
    parser.add_argument('--config_path', required=True, type=str,
                        help='Path to json file with all the parameters values')
    parser.add_argument('--custom_bottom_function_name', required=True, type=str,
                        help='Name of function from custom_models.py to use as bottom for model. '
                             'To this network ""survival" head will be appended')
    parser.add_argument('--model_type', required=False, type=str,
                        default='base', choices=['base', 'binary', 'contrastive'],
                        help='Defines loss function. If "base", log-likelihood is optimized.'
                             'If "binary", binary cross-entropy is added to loss function. '
                             'If "contrastive", contrastive loss is added to loss function.')
    parser.add_argument('--verbose', required=False, type=int, choices=[0, 1], default=0,
                        help='Whether to print detailed stats')
    # saving options
    parser.add_argument('--save_path', required=False, type=str, default='tmp/',
                        help='Path to store data in case of save_model or save_prediction options is on')
    parser.add_argument('--save_model', required=False, type=bool, choices=[True, False],
                        default=False, help='Whether to save model to `save_path`_model.pkl')
    parser.add_argument('--save_prediction', required=False, type=bool, choices=[True, False],
                        default=False, help='Whether to save predictions to `save_path`_pred.pkl')
    args = vars(parser.parse_args())
    print("Arguments: ")
    print(args)
    main(args)
    # TODO: before testing save preprocessed datasets
