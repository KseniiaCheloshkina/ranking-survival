import os
import copy
import pickle
import pandas as pd
import numpy as np
import json
from tabulate import tabulate
import tensorflow as tf

import train
from tools import test_quality


def eval():
    # print("----------------------------------------->")
    # print("Train model and evaluate for KKBOX dataset")
    # args, base_config_path, bin_config_path, contr_config_path, name = init_kkbox()
    # all_res = []
    # # Train base model
    # print("Start training base model...")
    # model_type = "base"
    # args_base = copy.deepcopy(args)
    # args_base.update({
    #     'model_type': model_type,
    #     'config_path': base_config_path
    # })
    # args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in args_base.items()])
    # os.system('python train.py {}'.format(args_str))
    # # evaluate results
    # prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    # with open(args_base['config_path'], 'rb') as f:
    #     config = json.load(f)
    # df_quality = calc_stats(args_base, config, prediction_path)
    # df_quality['model_type'] = model_type
    # all_res.append(df_quality)
    #
    # # TODO: add cross-entropy and contrastive models
    # # bin_config_path
    # # contr_config_path
    #
    # # save final results
    # df = pd.concat(all_res)
    # with open(args['save_path'] + "eval_metrics_" + name + ".pkl", 'wb') as f:
    #     pickle.dump(df, f)
    # res_kkbox = df.sort_values('epoch').tail(1)
    # res_kkbox['dataset'] = name

    print("----------------------------------------->")
    print("Train model and evaluate for METABRIC dataset")
    args, base_config_path, bin_config_path, contr_config_path, name = init_metabric()
    all_res = []

    # Train base model
    df_quality = train_one_model(args, config_path=base_config_path, model_type="base")
    all_res.append(df_quality)

    # Train binary model
    # df_quality = train_one_model(args, config_path=bin_config_path, model_type="binary")
    # all_res.append(df_quality)

    # Train contrastive model
    df_quality = train_one_model(args, config_path=contr_config_path, model_type="contrastive")
    all_res.append(df_quality)

    # save final results
    df = pd.concat(all_res)
    with open(args['save_path'] + "eval_metrics_" + name + ".pkl", 'wb') as f:
        pickle.dump(df, f)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'epoch'})
    res_metabric = df.sort_values('epoch')
    res_metabric['dataset'] = name
    return res_metabric

    # df_final = pd.concat([res_kkbox, res_metabric])
    # return df_final


def train_one_model(args, config_path, model_type):
    # Train binary model
    print("Start training {} model...".format(model_type))
    args_base = copy.deepcopy(args)
    args_base.update({
        'model_type': model_type,
        'config_path': config_path
    })
    args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in args_base.items()])
    print(args_str)
    os.system('python3.6 train.py {}'.format(args_str))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "_val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base, config, prediction_path)
    df_quality['model_type'] = model_type
    return df_quality


def init_kkbox():
    name = 'kkbox'
    # define data and config
    args = dict(
        train_data_path="data/kkbox_preprocessed_train.pkl",
        val_data_path="data/kkbox_preprocessed_test.pkl",
        custom_bottom_function_name="kkbox_main_network",
        verbose=1,
        save_path="data/reproduce_kkbox/",
        save_prediction=True,
        save_losses=True
    )
    # define configs
    base_config_path = "configs/config_kkbox_base.json"
    bin_config_path = "configs/config_kkbox_binary.json"
    contr_config_path = "configs/config_kkbox_contrastive.json"
    return args, base_config_path, bin_config_path, contr_config_path, name


def init_metabric():
    name = 'metabric'
    # define data and config
    args = dict(
        train_data_path="data/metabric/metabric_preprocessed_cv_0_train.pkl",
        val_data_path="data/metabric/metabric_preprocessed_cv_0_test.pkl",
        custom_bottom_function_name="metabric_main_network",
        verbose=1,
        save_path="data/reproduce_metabric/",
        save_prediction=True,
        save_losses=True
    )
    # define configs
    base_config_path = "configs/config_metabric_base.json"
    bin_config_path = "configs/config_metabric_binary.json"
    contr_config_path = "configs/config_metabric_contrastive.json"
    return args, base_config_path, bin_config_path, contr_config_path, name


def calc_stats_old(args, config, prediction_path):
    with open(prediction_path, 'rb') as f:
        pred_val = pickle.load(f)
    with open(args['val_data_path'], 'rb') as f:
        data = pickle.load(f)
    all_q = []
    # evaluate quality every 3 epochs
    every_nth_epoch = 3
    pred_to_estimate = pred_val[::every_nth_epoch]
    for idx, pred in enumerate(pred_to_estimate):
        q = test_quality(t_true=data['t'], y_true=data['y'], pred=pred,
                         time_grid=config['time_grid'], concordance_at_t=None, plot=False)
        q['epoch'] = every_nth_epoch * idx
        all_q.append(q)
    q = pd.concat(all_q)
    # load model and data
    with open(args['train_data_path'], 'rb') as f:
        train_data = pickle.load(f)
    with open(args['val_data_path'], 'rb') as f:
        val_data = pickle.load(f)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        # Restore variables from disk
        saver = tf.train.import_meta_graph(args['save_path'] + args["model_type"] + "_model" + '.meta')
        saver.restore(sess, args['save_path'] + args["model_type"] + "_model")
        with open(args['save_path'] + args["model_type"] + "_model.pkl", 'rb') as f:
            model = pickle.load(f)
        # evaluate loss
        train_main_loss, train_loss = train.get_loss_batch(train_data, config, sess, [model.main_loss, model.loss])
        val_main_loss, val_loss = train.get_loss_batch(val_data, config, sess, [model.main_loss, model.loss])
    q['train_main_loss'] = ''
    q.at[0, 'train_main_loss'] = [train_main_loss]
    q['train_loss'] = ''
    q.at[0, 'train_loss'] = [train_loss]
    q['val_main_loss'] = ''
    q.at[0, 'val_main_loss'] = [val_main_loss]
    q['val_loss'] = ''
    q.at[0, 'val_loss'] = [val_loss]
    q.reset_index(drop=True, inplace=True)
    return q


def calc_stats(args, config, prediction_path):
    with open(prediction_path, 'rb') as f:
        pred_val = pickle.load(f)
    with open(args['val_data_path'], 'rb') as f:
        data = pickle.load(f)
    df_losses = pd.read_csv(args['save_path'] + args["model_type"] + '_losses.csv', index_col=0)
    q = []
    for pred in pred_val:
        q.append(
            test_quality(
                t_true=data['t'], y_true=data['y'], pred=pred, time_grid=np.array(config['time_grid']),
                concordance_at_t=None, plot=False
            )
        )
    df_all_q = pd.concat(q)
    df_all_q.reset_index(drop=True, inplace=True)
    df_all_q = pd.concat([df_losses, df_all_q], axis=1)
    return df_all_q


if __name__ == "__main__":
    df_final = eval()
    df_final['rank'] = df_final.groupby(['dataset', 'model_type'])['epoch'].rank(ascending=False)
    df_final = df_final[df_final['rank'] == 1].drop(['rank'], axis=1)
    print(tabulate(df_final, headers=df_final.columns))
