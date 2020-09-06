import os
import copy
import pickle
import pandas as pd
import json
from tabulate import tabulate

import train
from tools import test_quality


def eval():
    print("----------------------------------------->")
    print("Train model and evaluate for KKBOX dataset")
    args, base_config_path, bin_config_path, contr_config_path, name = init_kkbox()
    all_res = []
    # Train base model
    print("Start training base model...")
    model_type = "base"
    args_base = copy.deepcopy(args)
    args_base.update({
        'model_type': model_type,
        'config_path': base_config_path
    })
    args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in args_base.items()])
    os.system('python train.py {}'.format(args_str))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base['val_data_path'], prediction_path, config['time_grid'])
    df_quality['model_type'] = model_type
    all_res.append(df_quality)

    # TODO: add cross-entropy and contrastive models
    # bin_config_path
    # contr_config_path

    # save final results
    df = pd.concat(all_res)
    with open(args['save_path'] + "eval_metrics_" + name + ".pkl", 'wb') as f:
        pickle.dump(df, f)
    res_kkbox = df.sort_values('epoch').tail(1)
    res_kkbox['dataset'] = name

    print("----------------------------------------->")
    print("Train model and evaluate for METABRIC dataset")
    args, base_config_path, bin_config_path, contr_config_path = init_metabric()
    all_res = []
    # Train base model
    print("Start training base model...")
    model_type = "base"
    args_base = copy.deepcopy(args)
    args_base.update({
        'model_type': model_type,
        'config_path': base_config_path
    })
    args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in args_base.items()])
    os.system('python train.py {}'.format(args_str))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base['val_data_path'], prediction_path, config['time_grid'])
    df_quality['model_type'] = model_type
    all_res.append(df_quality)

    # TODO: add cross-entropy and contrastive models
    # bin_config_path
    # contr_config_path

    # save final results
    df = pd.concat(all_res)
    with open(args['save_path'] + "eval_metrics_" + name + ".pkl", 'wb') as f:
        pickle.dump(df, f)
    res_metabric = df.sort_values('epoch').tail(1)
    res_metabric['dataset'] = name
    df_final = pd.concat([res_kkbox, res_metabric])
    return df_final


def init_kkbox():
    name = 'kkbox'
    # define data and config
    args = dict(
        train_data_path="data/kkbox_preprocessed_train.pkl",
        val_data_path="data/kkbox_preprocessed_test.pkl",
        custom_bottom_function_name="kkbox_main_network",
        verbose=1,
        save_path="data/reproduce_kkbox/",
        save_model=True,
        save_prediction=True,
    )
    # define configs
    base_config_path = "data/config_kkbox_base.json"
    bin_config_path = "data/config_kkbox_binary.json"
    contr_config_path = "data/config_kkbox_contrastive.json"
    return args, base_config_path, bin_config_path, contr_config_path, name


def init_metabric():
    name = 'metabric'
    # define data and config
    args = dict(
        train_data_path="data/metabric_preprocessed_train.pkl",
        val_data_path="data/metabric_preprocessed_test.pkl",
        custom_bottom_function_name="metabric_main_network",
        verbose=1,
        save_path="data/reproduce_metabric/",
        save_model=True,
        save_prediction=True,
    )
    # define configs
    base_config_path = "data/config_metabric_base.json"
    bin_config_path = "data/config_metabric_binary.json"
    contr_config_path = "data/config_metabric_contrastive.json"
    return args, base_config_path, bin_config_path, contr_config_path, name


def calc_stats(data_path, prediction_path, time_grid):
    with open(prediction_path, 'rb') as f:
        pred_val = pickle.load(f)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    all_q = []
    # evaluate quality every 3 epochs
    every_nth_epoch = 3
    pred_to_estimate = pred_val[::every_nth_epoch]
    for idx, pred in enumerate(pred_to_estimate):
        q = test_quality(t_true=data['t'], y_true=data['y'], pred=pred,
                         time_grid=time_grid, concordance_at_t=None, plot=False)
        q['epoch'] = every_nth_epoch * idx
        all_q.append(q)
    q = pd.concat(all_q)
    q.reset_index(drop=True, inplace=True)
    # TODO: load model and evaluate loss
    # q['test_loss'] = ''
    # q['train_loss'] = ''
    # q.at[0, 'test_loss'] = val_loss
    # q.at[0, 'train_loss'] = train_loss
    return q


if __name__ == "__main__":
    df_final = eval()
    print(tabulate(df_final))
