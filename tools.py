import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt


def test_quality(t_true, y_true, pred,
                 time_grid=np.linspace(0, 300, 30, dtype=np.int),
                 concordance_at_t=None):
    # get survival proba for time_grid
    all_surv_time = pd.DataFrame()
    for t in time_grid:
        surv_prob = np.exp(-1 * np.power(t / (pred[:, 0] + 1e-6), pred[:, 1]))
        all_surv_time = pd.concat([all_surv_time, pd.DataFrame(surv_prob).T])
    all_surv_time.index = time_grid

    if concordance_at_t is None:
        concordance_at_t = np.mean(time_grid)
    harell_c_index = concordance_index(
        predicted_scores=all_surv_time.loc[concordance_at_t, :].values,
        event_times=t_true,
        event_observed=y_true)

    ev = EvalSurv(surv=all_surv_time, durations=t_true, events=y_true,
                  censor_surv='km')
    dt_c_index = ev.concordance_td('antolini')
    int_brier_score = ev.integrated_brier_score(time_grid)
    int_nbill = ev.integrated_nbll(time_grid)

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

    return pd.DataFrame([
        {
            'harell_c_index': harell_c_index,
            'dt_c_index': dt_c_index,
            'int_brier_score': int_brier_score,
            'int_nbill': int_nbill
        }
    ])
