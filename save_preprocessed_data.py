import pickle

import pandas as pd
import numpy as np
from pycox.datasets import metabric, kkbox_v1
from sklearn.model_selection import StratifiedKFold

from tools import preprocess_kkbox, transform_kkbox


def get_metabric():
    df_all = metabric.read_df()
    df_all = df_all[df_all['duration'] != 0]
    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    quan = np.arange(0, 1, 0.2)
    bins = df_all['duration'].quantile(quan).values
    bins = np.append(bins, np.inf)
    df_all['duration_bin'] = pd.cut(df_all['duration'], bins=bins, labels=np.arange(0, 5, 1))
    df_all['strata'] = df_all['event'].astype(str) + "_" + df_all['duration_bin'].astype(int).astype(str)
    df_all = df_all[df_all['strata'] != '0_-9223372036854775808']

    # get cv data
    df_all['duration'] = df_all['duration'].astype(int)
    df_all = df_all[df_all['duration'] != 0]
    df_all.reset_index(drop=True, inplace=True)
    sf = StratifiedKFold(n_splits=5, random_state=1)
    df_cv_data = dict()
    for i, (tr_i, te_i) in enumerate(sf.split(df_all, df_all['strata'])):
        # split
        df_train = df_all.iloc[tr_i, ]
        df_test = df_all.iloc[te_i, ]
        # preprocess
        m = df_train[cols_standardize].mean()
        v = df_train[cols_standardize].std()
        for idx, col in enumerate(cols_standardize):
            df_train[col] = df_train[col] - m.values[idx]
            df_train[col] = df_train[col] / v.values[idx]
            df_test[col] = df_test[col] - m.values[idx]
            df_test[col] = df_test[col] / v.values[idx]
        df_cv_data[i] = {
            'test': {
                'x': df_test[cols_standardize + cols_leave].values,
                'y': df_test['event'].values,
                't': df_test['duration'].astype(int).values
            },
            'train': {
                'x': df_train[cols_standardize + cols_leave].values,
                'y': df_train['event'].values,
                't': df_train['duration'].astype(int).values
            }
        }
    for i in range(5):
        with open('data/input_metabric/metabric_preprocessed_cv_{}_train.pkl'.format(i), 'wb') as f:
            pickle.dump(df_cv_data[i]['train'], f)
        with open('data/input_metabric/metabric_preprocessed_cv_{}_test.pkl'.format(i), 'wb') as f:
            pickle.dump(df_cv_data[i]['test'], f)


def get_kkbox():
    df_train = kkbox_v1.read_df(subset='train')
    df_test = kkbox_v1.read_df(subset='test')
    df_val = kkbox_v1.read_df(subset='val')

    x_train, t_train, y_train = preprocess_kkbox(df_train)
    x_val, t_val, y_val = preprocess_kkbox(df_val)
    x_test, t_test, y_test = preprocess_kkbox(df_test)

    x_train, x_test, x_val = transform_kkbox(x_train, x_test, x_val)

    data = {
        'x': x_train,
        'y': y_train,
        't': t_train
    }
    with open('data/input_kkbox/kkbox_preprocessed_train.pkl', 'wb') as f:
        pickle.dump(data, f)

    data = {
        'x': x_test,
        'y': y_test,
        't': t_test
    }
    with open('data/input_kkbox/kkbox_preprocessed_test.pkl', 'wb') as f:
        pickle.dump(data, f)

    data = {
        'x': x_val,
        'y': y_val,
        't': t_val
    }
    with open('data/input_kkbox/kkbox_preprocessed_val.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    print("Saving metabric...")
    get_metabric()
    print("Saving kkbox...")
    get_kkbox()
