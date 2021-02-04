import os
import json
import argparse
import copy
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt

from eval import calc_stats


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_config_vals(x):
    return pd.DataFrame([json.loads(x.replace("'", "\""))])


def get_summary(csv_path):
    data = pd.read_csv(csv_path, index_col=0)
    # results on last epoch
    data['rn'] = data.groupby(['config', 'val_path', 'train_path'])['epoch'].rank(ascending=False)
    last_epoch_data = data[data['rn'] == 1]
    last_epoch_data.drop(['epoch', 'rn'], axis=1, inplace=True)

    # parse config
    all_config_vals = []
    for idx, row in last_epoch_data.iterrows():
        all_config_vals.append(get_config_vals(row['config']))
    all_config_vals = pd.concat(all_config_vals, axis=0)
    config_params_names = list(all_config_vals.columns)
    last_epoch_data_final = pd.concat(
        [last_epoch_data.reset_index(drop=True), all_config_vals.reset_index(drop=True)], axis=1)

    # median values
    med_vals = (
        last_epoch_data_final
        .groupby(['config'])
        .agg({'dt_c_index': 'median', 'int_brier_score': 'median', 'int_nbill': 'median'})
        .sort_values(['dt_c_index', 'int_brier_score', 'int_nbill'], ascending=[False, True, True])
        .reset_index()
    )
    all_config_vals = []
    for idx, row in med_vals.iterrows():
        all_config_vals.append(get_config_vals(row['config']))
    all_config_vals = pd.concat(all_config_vals, axis=0)
    med_vals = pd.concat(
        [med_vals.reset_index(drop=True), all_config_vals.reset_index(drop=True)], axis=1)
    med_vals.drop(['config'], axis=1, inplace=True)
    print("Median values at last epoch :\n{}".format(tabulate(med_vals, headers=med_vals.columns)))
    print("Best configuration:\n{}".format(
        tabulate(med_vals.head(1)[config_params_names], headers=config_params_names)))

    # plot quality distribution by each changing hyperparameter
    save_path = "output.png"
    csv_paths = csv_path.split("/")
    if len(csv_paths) > 1:
        save_path = "/".join(csv_paths[:-1]) + "/output.png"
    changing_cols = {col: med_vals[col].nunique() for col in config_params_names
                     if not isinstance(med_vals[col].values[0], list)}
    changing_cols = [col for col, val in changing_cols.items() if val > 1]
    df = []
    for col in changing_cols:
        new_df = med_vals.copy(deep=True)
        new_df['changing_col_name'] = col
        new_df['changing_col_val'] = new_df[col]
        df.append(new_df)
    if df != list():
        df = pd.concat(df)
        metr_cols = ['dt_c_index', 'int_brier_score', 'int_nbill']
        df_melted = pd.melt(df, id_vars=['changing_col_val', 'changing_col_name'], value_vars=metr_cols,
                            var_name="metric_name", value_name="metric_val")
        sns_plot = sns.catplot(data=df_melted, x='changing_col_val', y='metric_val', col='changing_col_name',
                               row='metric_name', kind='box', sharey=False)
        sns_plot.savefig(save_path)
        print("Plot saved to {}".format(save_path))


def main(args):
    # generate list of configs
    with open(args['config_path'], 'rb') as f:
        config = json.load(f)
    tunable_params = [
        'alpha_reg', 'alpha_bias_random_mean', 'alpha_random_stddev', 'beta_random_stddev', 'n_time_bins', 'n_ex_bin',
        'n_epochs', 'step_rate', 'decay', 'learning_rate', 'optimizer',
        'cross_entropy_weight', 'margin_weight', 'contrastive_weight',
        'n_epochs_contr', 'learning_rate_contr', 'step_rate_contr', 'decay_contr', 'momentum_contr', 'optimizer_contr',
        'n_epochs_both', 'learning_rate_both', 'step_rate_both', 'decay_both', 'momentum_both', 'optimizer_both'
    ]
    grid = {k: v if isinstance(v, list) else [v] for k, v in config.items() if k in tunable_params}
    grid = list(ParameterGrid(grid))
    full_grid = []
    for conf in grid:
        conf.update({k: v for k, v in config.items() if k not in tunable_params})
        full_grid.append(conf)

    # for each config for each train-val pair train model
    train_args = {arg_key: arg_value for arg_key, arg_value in args.items() if
                  arg_key not in ['train_data_paths', 'val_data_paths', 'config_path']}
    train_args.update({'config_path': 'tmp_config.json'})
    all_res = []
    printProgressBar(0, len(full_grid), prefix='Progress:', suffix='Complete', length=50)
    for pr_bar_num, config in enumerate(full_grid):
        with open("tmp_config.json", 'w') as f:
            json.dump(config, f)
        current_train_args = copy.deepcopy(train_args)
        for train_path, val_path in zip(args['train_data_paths'], args['val_data_paths']):
            current_train_args.update({'train_data_path': train_path, 'val_data_path': val_path})
            args_str = " ".join(
                ["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in current_train_args.items()])
            print(args_str)
            # train
            os.system('python3.7 train.py {}'.format(args_str))
            # evaluate results
            prediction_path = current_train_args['save_path'] + current_train_args['model_type'] + "_val_pred.pkl"
            df_quality = calc_stats(current_train_args, config, prediction_path)
            df_quality['config'] = str(config)
            df_quality['val_path'] = val_path
            df_quality['train_path'] = train_path
            col_list = df_quality.columns
            new_col_list = ['epoch']
            new_col_list.extend(col_list)
            df_quality.reset_index(inplace=True)
            df_quality.columns = new_col_list
            all_res.append(df_quality)
            # save after each run
            pd.concat(all_res).to_csv(current_train_args['save_path'] + "eval_metrics.csv")
            # Update Progress Bar
            printProgressBar(pr_bar_num, len(full_grid), prefix='Progress:', suffix='Complete', length=50)
    # analyze results
    get_summary(current_train_args['save_path'] + "eval_metrics.csv")
    plot_results(current_train_args['save_path'] + "eval_metrics.csv")


def plot_results(path):
    df = pd.read_csv(path, index_col=0)
    all_config_vals = []
    for idx, row in df.iterrows():
        all_config_vals.append(get_config_vals(row['config']))
    all_config_vals = pd.concat(all_config_vals, axis=0)
    config_params_names = list(all_config_vals.columns)
    changing_cols = {col: all_config_vals[col].nunique() for col in config_params_names
                     if not isinstance(all_config_vals[col].values[0], list)}
    changing_cols = [col for col, val in changing_cols.items() if val > 1]
    if changing_cols == list():
        df['config'] = 'single config'
    else:
        df['config'] = [str(params) for params in all_config_vals[changing_cols].to_dict('records')]
    df_melted = pd.melt(df, id_vars=['config', 'epoch', 'val_path', 'train_path'])
    df_melted.drop(df_melted.loc[(df_melted['variable'] == 'val_loss') & (df_melted['epoch'] < 28)].index, axis=0, inplace=True)
    df_melted.drop(df_melted.loc[(df_melted['variable'] == 'train_loss') & (df_melted['epoch'] < 28)].index, axis=0,
                   inplace=True)
    sns_plot = sns.catplot(data=df_melted, hue='config', x='epoch', y='value', row='variable', sharey=False, kind='point')
    sns_plot.savefig(path.replace(".csv", ""))


if __name__ == "__main__":
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
