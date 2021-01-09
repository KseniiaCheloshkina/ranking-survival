import os
import json
import argparse
import copy
from sklearn.model_selection import ParameterGrid
import pandas as pd

from eval import calc_stats


def main(args):
    # generate list of configs
    with open(args['config_path'], 'rb') as f:
        config = json.load(f)
    base_tunable_params = [
        'alpha_reg',  'alpha_bias_random_mean', 'alpha_random_stddev', 'beta_random_stddev', 'n_time_bins', 'n_ex_bin',
        'n_epochs', 'step_rate', 'decay', 'learning_rate', 'optimizer']
    grid = {k: v if isinstance(v, list) else [v] for k, v in config.items() if k in base_tunable_params}
    grid = list(ParameterGrid(grid))
    full_grid = []
    for conf in grid:
        conf.update({k: v for k, v in config.items() if k not in base_tunable_params})
        full_grid.append(conf)

    # for each config for each train-val pair train model
    train_args = {arg_key: arg_value for arg_key, arg_value in args.items() if
                  arg_key not in ['train_data_paths', 'val_data_paths', 'config_path']}
    train_args.update({'config_path': 'tmp_config.json'})
    all_res = []
    for config in full_grid:
        with open("tmp_config.json", 'w') as f:
            json.dump(config, f)
        current_train_args = copy.deepcopy(train_args)
        for train_path, val_path in zip(args['train_data_paths'], args['val_data_paths']):
            current_train_args.update({'train_data_path': train_path, 'val_data_path': val_path})
            args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in current_train_args.items()])
            print(args_str)
            # train
            os.system('python3.7 train.py {}'.format(args_str))
            # evaluate results
            prediction_path = current_train_args['save_path'] + current_train_args['model_type'] + "_val_pred.pkl"
            df_quality = calc_stats(current_train_args, config, prediction_path)
            df_quality['config'] = str(config)
            df_quality['val_path'] = val_path
            df_quality['train_path'] = train_path
            all_res.append(df_quality)
    pd.concat(all_res).to_csv(current_train_args['save_path'] + "eval_metrics.csv")


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Train model on given dataset')
    # running options
    parser.add_argument('--train_data_paths', nargs="+", required=True, type=str,
                        help='''
                        LIST OF Paths to pickle files to train model on 
                        Example: --train_data_paths path1 path2 path3. 
                        Format of each pickle file: pickle file contains a dictionary with 3 keys:
                        x - features numpy array (n_samples, n_features)
                        y - event label numpy array of type int (n_samples, )
                        t - duration label numpy array of type int (n_samples, )
                        ''')
    parser.add_argument('--val_data_paths', nargs="+", required=False, type=str,
                        help='''
                        LIST OF Path to pickle files to validate model on
                        Example: --val_data_paths path1 path2 path3. 
                        Format of each pickle file: pickle file contains a dictionary with 3 keys:
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
                        help='Path to store data in case of save_prediction options is on')
    parser.add_argument('--save_prediction', required=False, type=bool, choices=[True, False],
                        default=False, help='Whether to save predictions to `save_path`/`model_type`_pred.pkl')
    parser.add_argument('--save_losses', required=False, type=bool, choices=[True, False],
                        default=False, help='Whether to save losses history to `save_path`/`model_type`_losses.csv')
    arguments = vars(parser.parse_args())
    print("Arguments: ")
    print(arguments)
    main(arguments)
