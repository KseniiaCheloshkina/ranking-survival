import os
import sys
import copy
import pickle
from time import time

import pandas as pd
import numpy as np
import json
from tabulate import tabulate

from tools import test_quality


def eval_kkbox(init_function_name="init_kkbox"):
    print("----------------------------------------->")
    print("Train model and evaluate for KKBOX dataset")
    init_function = getattr(sys.modules[__name__], init_function_name)
    args, base_config_path, bin_config_path, contr_config_path, name, report_path = init_function()
    all_res = []

    name_model = [
        ('base', base_config_path),
        ('binary', bin_config_path),
        ('contrastive', contr_config_path)
    ]

    for model_type, config_path in name_model:
        df_quality = train_one_model(args, config_path=config_path, model_type=model_type)
        all_res.append(df_quality)

    # save final results
    df = pd.concat(all_res)
    with open(report_path + "eval_metrics_" + name + ".pkl", 'wb') as f:
        pickle.dump(df, f)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'epoch'})
    res = df.sort_values('epoch')
    res['dataset'] = name
    res.to_csv(report_path + "report.csv")
    return res


def eval_metabric(init_function_name="init_metabric"):
    print("----------------------------------------->")
    print("Train model and evaluate for METABRIC dataset")
    init_function = getattr(sys.modules[__name__], init_function_name)
    all_res = []
    for i in range(5):
        args, base_config_path, bin_config_path, contr_config_path, name, report_path = init_function(i)
        name_model = [
            ('base', base_config_path),
            ('binary', bin_config_path),
            ('contrastive', contr_config_path)
            ]

        for model_type, config_path in name_model:
            df_quality = train_one_model(args, config_path=config_path, model_type=model_type)
            df_quality['cv'] = i
            all_res.append(df_quality)

    # save final results
    df = pd.concat(all_res)
    with open(report_path + "eval_metrics_" + name + ".pkl", 'wb') as f:
        pickle.dump(df, f)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'epoch'})
    res_metabric = df.sort_values('epoch')
    res_metabric['dataset'] = name
    res_metabric.to_csv(report_path + "report.csv")
    return res_metabric


def train_one_model(args, config_path, model_type):
    print("Start training {} model...".format(model_type))
    args_base = copy.deepcopy(args)
    args_base.update({
        'model_type': model_type,
        'config_path': config_path
    })
    args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in args_base.items()])
    print(args_str)
    start_time = time()
    os.system('python3.7 train.py {}'.format(args_str))
    train_time = time() - start_time
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "_val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base, config, prediction_path)
    df_quality['model_type'] = model_type
    df_quality['time'] = train_time
    return df_quality


def init_kkbox():
    name = 'kkbox'
    # define data and config
    args = dict(
        train_data_path="data/input_kkbox/kkbox_preprocessed_train.pkl",
        val_data_path="data/input_kkbox/kkbox_preprocessed_test.pkl",
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
    report_path = "data/reproduce_kkbox/"
    return args, base_config_path, bin_config_path, contr_config_path, name, report_path


def init_metabric(cv=0):
    name = 'metabric'
    # define data and config
    args = dict(
        train_data_path="data/input_metabric/metabric_preprocessed_cv_{}_train.pkl".format(cv),
        val_data_path="data/input_metabric/metabric_preprocessed_cv_{}_test.pkl".format(cv),
        custom_bottom_function_name="metabric_main_network",
        verbose=1,
        save_path="data/reproduce_metabric/fold_{}/".format(cv),
        save_prediction=True,
        save_losses=True
    )
    # define configs
    base_config_path = "configs/config_metabric_base.json"
    bin_config_path = "configs/config_metabric_binary.json"
    contr_config_path = "configs/config_metabric_contrastive.json"
    report_path = "data/reproduce_metabric/"
    return args, base_config_path, bin_config_path, contr_config_path, name, report_path


def init_metabric_benchmark(cv=0):
    name = 'metabric'
    # define data and config
    args = dict(
        train_data_path="data/input_metabric/metabric_preprocessed_cv_{}_train.pkl".format(cv),
        val_data_path="data/input_metabric/metabric_preprocessed_cv_{}_test.pkl".format(cv),
        custom_bottom_function_name="metabric_main_network",
        verbose=1,
        save_path="data/reproduce_metabric/fold_{}/".format(cv),
        save_prediction=True,
        save_losses=True
    )
    # define configs
    base_config_path = "configs/config_metabric_base_bench.json"
    bin_config_path = "configs/config_metabric_binary_bench.json"
    contr_config_path = "configs/config_metabric_contrastive_bench.json"
    report_path = "data/benchmark_metabric/"
    return args, base_config_path, bin_config_path, contr_config_path, name, report_path


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
    res_metabric = eval_metabric()
    res_kkbox = eval_kkbox()
    res_kkbox['cv'] = 'test'
    df_final = pd.concat([res_kkbox, res_metabric])
    df_final = res_kkbox
    df_final['rank'] = df_final.groupby(['dataset', 'model_type', 'cv'])['epoch'].rank(ascending=False)
    df_final = df_final[df_final['rank'] == 1].drop(['rank'], axis=1)
    df_final.drop(['epoch'], axis=1, inplace=True)
    df_final = df_final.groupby(['dataset', 'model_type']).agg({
        'train_loss': 'mean', 'train_main_loss': 'mean', 'val_loss': 'mean', 'val_main_loss': 'mean',
        'dt_c_index': 'mean', 'int_brier_score': 'mean', 'int_nbill': 'mean', 'time': 'mean'
    })
    print(tabulate(df_final, headers=df_final.columns))
