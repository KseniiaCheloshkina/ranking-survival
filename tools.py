import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler


def test_quality(t_true, y_true, pred,
                 time_grid=np.linspace(0, 300, 30, dtype=np.int),
                 concordance_at_t=None, plot=False):
    # get survival proba for time_grid
    all_surv_time = pd.DataFrame()
    for t in time_grid:
        surv_prob = np.exp(-1 * np.power(t / (pred[:, 0] + 1e-6), pred[:, 1]))
        all_surv_time = pd.concat([all_surv_time, pd.DataFrame(surv_prob).T])
    all_surv_time.index = time_grid

    ev = EvalSurv(surv=all_surv_time, durations=t_true, events=y_true,
                  censor_surv='km')
    dt_c_index = ev.concordance_td('antolini')
    int_brier_score = ev.integrated_brier_score(time_grid)
    int_nbill = ev.integrated_nbll(time_grid)

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(20, 7))
        d = all_surv_time.sample(5, axis=1).loc[1:]
        obs = d.columns
        for o in obs:
            ax[0].plot(d.index, d[o])
        ax[0].set_xlabel('Time')
        ax[0].set_title("Sample survival curves")
        nb = ev.nbll(time_grid)
        ax[1].plot(time_grid, nb)
        ax[1].set_title('NBLL')
        ax[1].set_xlabel('Time')
        br = ev.brier_score(time_grid)
        ax[2].plot(time_grid, br)
        ax[2].set_title('Brier score')
        ax[2].set_xlabel('Time')
        plt.show();
        
    if concordance_at_t is not None:
        harell_c_index = concordance_index(
            predicted_scores=all_surv_time.loc[concordance_at_t, :].values,
            event_times=t_true,
            event_observed=y_true)
        
        return pd.DataFrame([
            {
                'harell_c_index': harell_c_index,
                'dt_c_index': dt_c_index,
                'int_brier_score': int_brier_score,
                'int_nbill': int_nbill
            }
        ])
    else:
        return pd.DataFrame([
            {
                'dt_c_index': dt_c_index,
                'int_brier_score': int_brier_score,
                'int_nbill': int_nbill
            }
        ])


def preprocess_kkbox(df):
    def get_one_hot_encoded(values, cat_series):
        out_array = np.zeros((cat_series.shape[0], len(values)))
        for val in values:
            out_array[:, val] = (cat_series == val).astype(int)
        return out_array

    df1 = df.copy(deep=True)
    # target
    t = df['duration'].values
    y = df['event'].astype(int).values
    # work with cat features
    gender_map = {
        'male': 0,
        'female': 1
    }
    df1['gender'] = df['gender'].astype(str).map(gender_map).fillna(0).astype(int)

    cities_map = {
        '1.0': 0,
        '10.0': 1,
        '11.0': 2,
        '12.0': 3,
        '13.0': 4,
        '14.0': 5,
        '15.0': 6,
        '16.0': 7,
        '17.0': 8,
        '18.0': 9,
        '19.0': 10,
        '20.0': 11,
        '21.0': 12,
        '22.0': 13,
        '3.0': 14,
        '4.0': 15,
        '5.0': 16,
        '6.0': 17,
        '7.0': 18,
        '8.0': 19,
        '9.0': 20
    }
    df1['city'] = df['city'].astype(str).map(cities_map)

    registered_via_cats_map = {
        '9.0': 0,
        '7.0': 1,
        '3.0': 2,
        '4.0': 3,
        '13.0': 4
    }
    df1['registered_via'] = df['registered_via'].astype(float).astype(str).map(registered_via_cats_map)
    # categories to one-hot
    gender_values = [0, 1]
    city_values = np.arange(21)
    registered_via_values = np.arange(5)
    gender_ = get_one_hot_encoded(gender_values, df1['gender'])
    city_ = get_one_hot_encoded(city_values, df1['city'])
    registered_via_values_ = get_one_hot_encoded(registered_via_values, df1['registered_via'])
    # binary cols as is
    binary_cols = ['is_auto_renew', 'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns']
    binary_ = df1[binary_cols].values
    # continous cols  - apply batchnorm in network
    batchnorm_cols = ['n_prev_churns', 'log_days_between_subs', 'log_days_since_reg_init', 'log_payment_plan_days',
                      'log_plan_list_price',
                      'log_actual_amount_paid', 'age_at_start']
    batchnorm_ = df1[batchnorm_cols].values
    # final features
    x = np.hstack([gender_, city_, registered_via_values_, binary_, batchnorm_])
    return x, t, y


def transform_kkbox(x_train, x_test, x_val):
    cols_to_scale = [i for i in range(33, 40)]
    # fit scaler
    st_sc = StandardScaler().fit(X=x_train[:, cols_to_scale])
    # transform
    x_train[:, cols_to_scale] = st_sc.transform(x_train[:, cols_to_scale])
    x_test[:, cols_to_scale] = st_sc.transform(x_test[:, cols_to_scale])
    x_val[:, cols_to_scale] = st_sc.transform(x_val[:, cols_to_scale])
    return x_train, x_test, x_val
