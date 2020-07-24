import os
import copy
import pickle
import pandas as pd
import json

import train
from tools import test_quality


def eval_kkbox():
    # define data and config
    args = dict(
        train_data_path="data/kkbox_preprocessed_train.pkl",
        val_data_path="data/kkbox_preprocessed_test.pkl",
        custom_bottom_function_name="kkbox_main_network",
        verbose=0,
        save_path="reproduce_kkbox/",
        save_model=True,
        save_prediction=True,
        config_path="data/config_kkbox.json"
    )
    all_res = []

    # Train base model
    print("Start training base model...")
    model_type = "base"
    args_base = copy.deepcopy(args)
    args_base.update({'model_type': model_type})
    # os.command(train(args_base))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base['val_data_path'], prediction_path, config['time_grid'])
    df_quality['model_type'] = model_type
    all_res.append(df_quality)

    # Train cross-entropy model
    print("Start training cross-entropy model...")
    model_type = "binary"
    args_binary = copy.deepcopy(args)
    args_binary.update({'model_type': model_type})
    # os.command(train(args_binary))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base['val_data_path'], prediction_path, config['time_grid'])
    df_quality['model_type'] = model_type
    all_res.append(df_quality)

    # Train contrastive model
    print("Start training contrastive model...")
    model_type = "contrastive"
    args_contrastive = copy.deepcopy(args)
    args_contrastive.update({'model_type': model_type})
    # os.command(train(args_contrastive))
    # evaluate results
    prediction_path = args_base['save_path'] + model_type + "val_pred.pkl"
    with open(args_base['config_path'], 'rb') as f:
        config = json.load(f)
    df_quality = calc_stats(args_base['val_data_path'], prediction_path, config['time_grid'])
    df_quality['model_type'] = model_type
    all_res.append(df_quality)

    # save final results
    df = pd.concat(all_res)
    with open(args['save_path'] + "eval_metrics.pkl", 'wb') as f:
        pickle.dump(df, f)
    return df.sort_values('epoch').tail(1)


def eval_metabric():
    pass


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
    # q['test_loss'] = ''
    # q['train_loss'] = ''
    # q.at[0, 'test_loss'] = val_loss
    # q.at[0, 'train_loss'] = train_loss
    return q


if __name__ == "__main__":
    print("----------------------------------------->")
    print("Train model and evaluate for KKBOX dataset")
    df_kkbox = eval_kkbox()
    df_kkbox['dataset'] = 'KKBOX'
    print("----------------------------------------->")
    print("Train model and evaluate for METABRIC dataset")
    df_metabric = eval_metabric()
    df_metabric['dataset'] = 'METABRIC'

    df_final = pd.concat([df_metabric, df_kkbox])
    # TODO: print(tabulate(df_final))
