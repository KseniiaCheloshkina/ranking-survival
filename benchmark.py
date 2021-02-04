import numpy as np
import torch
from torch import nn
import torchtuples as tt
from pycox.evaluation import EvalSurv
import pycox
from pycox.models import DeepHitSingle
import pickle
import pandas as pd
import tabulate
import tqdm
from time import time

from eval import eval_metabric


class MetabricMainNetworkTorch(nn.Module):
    def __init__(self, in_features, out_features=4, last_layer_units=1, bias=True,
                 w_init_=lambda w: nn.init.xavier_normal(w)):
        super().__init__()
        self.linear_first = nn.Linear(in_features, out_features, bias)
        self.linear_output = nn.Linear(out_features, last_layer_units, bias)
        if w_init_:
            w_init_(self.linear_first.weight.data)
            w_init_(self.linear_output.weight.data)

    def forward(self, input):
        projection = self.linear_first(input)
        output = self.linear_output(projection)
        return output


class KKBOXMainNetworkTorch(nn.Module):
    def __init__(self, in_features, out_features=[64, 32, 16], last_layer_units=1, bias=True,
                 w_init_=lambda w: nn.init.xavier_normal(w)):
        super().__init__()
        self.gender_embed = nn.Linear(in_features=2, out_features=1, bias=True).cuda()
        self.city_embed = nn.Linear(in_features=21, out_features=4, bias=True).cuda()
        self.reg_embed = nn.Linear(in_features=5, out_features=2, bias=True).cuda()

        dense_layers = []
        prev_units = 19
        for n_units in out_features:
            layer = nn.Linear(prev_units, n_units, bias).cuda()
            if w_init_:
                w_init_(layer.weight.data)
            dense_layers.append(layer)
            dense_layers.append(nn.ReLU().cuda())
            prev_units = n_units
        self.dense_layers = dense_layers
        self.linear_output = nn.Linear(out_features[-1], last_layer_units, bias).cuda()
        if w_init_:
            w_init_(self.linear_output.weight.data)

    def forward(self, input):
        projection = torch.cat(
            (
                self.gender_embed(input[:, 0:2].float()),
                self.city_embed(input[:, 2:23].float()),
                self.reg_embed(input[:, 23:28].float()),
                input[:, 28:].float()
            ),
            1
        )
        for layer in self.dense_layers:
            projection = layer.forward(projection)
        output = self.linear_output(projection.float())
        return output


def benchmark_metabric():

    time_grid = np.linspace(0, 300, 30, dtype=np.int)
    get_target = lambda data: (data['t'], data['y'])
    # define base network
    in_features = 9
    epochs = 512
    net = MetabricMainNetworkTorch(in_features=in_features, out_features=4, last_layer_units=1, bias=True)
    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)
    deephit_net = MetabricMainNetworkTorch(in_features=in_features, out_features=4, last_layer_units=num_durations,
                                           bias=True)
    all_results = []

    # load data
    for cv in tqdm.tqdm(range(5)):
        train_data_path = "data/input_metabric/metabric_preprocessed_cv_{}_train.pkl".format(cv)
        val_data_path = "data/input_metabric/metabric_preprocessed_cv_{}_test.pkl".format(cv)
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_data_path, 'rb') as f:
            val_data = pickle.load(f)
        y_train = get_target(train_data)

        # CoxPH, CoxCC
        for method in ['CoxPH', 'CoxCC']:
            f = getattr(pycox.models, method)
            model = f(net, tt.optim.Adam)
            df_metrics = eval(model, train_data, y_train, val_data, time_grid, epochs=epochs, lr=0.01)
            df_metrics['cv'] = cv
            df_metrics['method'] = method
            all_results.append(df_metrics)

        # DeepHit method
        y_train = labtrans.fit_transform(*y_train)
        model = DeepHitSingle(deephit_net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
        df_metrics = eval(model, train_data, y_train, val_data, time_grid, epochs=epochs, lr=0.01)
        df_metrics['cv'] = cv
        df_metrics['method'] = 'DeepHit'
        all_results.append(df_metrics)
    return pd.concat(all_results, axis=0)


def benchmark_kkbox():

    time_grid = np.array([
        1,  29,  57,  85, 113, 142, 170, 198, 226, 255, 283, 311, 339,
        368, 396, 424, 452, 481, 509, 537, 565, 594, 622, 650, 678, 707,
        735, 763, 791, 820
    ])
    get_target = lambda data: (data['t'], data['y'])
    # define base network
    in_features = 40
    epochs = 70
    net = KKBOXMainNetworkTorch(in_features=in_features, out_features=[64, 32, 16], last_layer_units=1, bias=True,
                          w_init_=None)
    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)
    deephit_net = KKBOXMainNetworkTorch(in_features=in_features, out_features=[64, 32, 16],
                                        last_layer_units=num_durations, bias=True,
                                        w_init_=None)
    all_results = []

    # load data
    train_data_path = "data/input_kkbox/kkbox_preprocessed_train.pkl"
    val_data_path = "data/input_kkbox/kkbox_preprocessed_test.pkl"
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)
    y_train = get_target(train_data)

    # CoxPH, CoxCC
    for method in ['CoxPH', 'CoxCC']:
        f = getattr(pycox.models, method)
        model = f(net, tt.optim.Adam)
        df_metrics = eval(model, train_data, y_train, val_data, time_grid, epochs=epochs, lr=0.001)
        df_metrics['cv'] = 'test'
        df_metrics['method'] = method
        print(df_metrics)
        all_results.append(df_metrics)

    # DeepHit method
    y_train = labtrans.fit_transform(*y_train)
    model = DeepHitSingle(deephit_net, tt.optim.Adam, alpha=0.001, sigma=0.1, duration_index=labtrans.cuts)
    df_metrics = eval(model, train_data, y_train, val_data, time_grid, epochs=epochs, lr=0.001)
    df_metrics['cv'] = 'test'
    df_metrics['method'] = 'DeepHit'
    all_results.append(df_metrics)
    return pd.concat(all_results, axis=0)


def eval(model, train_data, y_train, val_data, time_grid, epochs=512, lr=0.001):
    batch_size = 256
    model.optimizer.set_lr(lr)
    start_time = time()
    _ = model.fit(train_data['x'], y_train, batch_size, epochs)
    end_time = time()
    try:
        _ = model.compute_baseline_hazards()
    except AttributeError:
        pass
    surv = model.predict_surv_df(val_data['x'])
    df_metrics = get_metrics(val_data, surv, time_grid)
    df_metrics['time'] = end_time - start_time
    return df_metrics


def get_metrics(val_data, surv_pred, time_grid):
    ev = EvalSurv(surv_pred, val_data['t'], val_data['y'], censor_surv='km')
    return pd.DataFrame([
        {
            'dt_c_index': ev.concordance_td('antolini'),
            'int_brier_score': ev.integrated_brier_score(time_grid),
            'int_nbill': ev.integrated_nbll(time_grid)
        }
    ])


if __name__ == "__main__":
    np.random.seed(1234)
    _ = torch.manual_seed(123)

    # METABRIC

    # run benchmark models
    df_res = benchmark_metabric()
    # save results
    with open("output/metabric_benchmark.pkl", 'wb') as f:
        pickle.dump(df_res, f)
    # mean
    cols = ["dt_c_index", "int_brier_score", "int_nbill", "time"]
    print("Benchmark methods: ")
    print(tabulate.tabulate(df_res.groupby(['method'])[cols].mean(), headers=cols))

    # run proposed models for comparable with benchmark time
    # results are saved to data/benchmark_metabric
    df_proposed = eval_metabric(init_function_name="init_metabric_benchmark")
    df_proposed['rank'] = df_proposed.groupby(['dataset', 'model_type', 'cv'])['epoch'].rank(ascending=False)
    df_proposed = df_proposed[df_proposed['rank'] == 1].drop(['rank'], axis=1)
    df_proposed.drop(['epoch'], axis=1, inplace=True)
    df_proposed = df_proposed.groupby(['dataset', 'model_type']).agg({
        'dt_c_index': 'mean', 'int_brier_score': 'mean', 'int_nbill': 'mean', 'time': 'mean'
    })
    print("Proposed methods: ")
    print(tabulate.tabulate(df_proposed, headers=cols))

    # KKBOX
    # run benchmark models
    df_res = benchmark_kkbox()
    # save results
    with open("output/kkbox_benchmark.pkl", 'wb') as f:
        pickle.dump(df_res, f)
    # mean
    cols = ["dt_c_index", "int_brier_score", "int_nbill", "time"]
    print("Benchmark methods: ")
    print(tabulate.tabulate(df_res.groupby(['method'])[cols].mean(), headers=cols))
