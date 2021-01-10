import pickle
import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), "data"))
sys.path.append(os.getcwd())

from batch_generators_hard_mining import get_valid_pairs_tf, get_valid_pairs, DataGenerator


def test_bg():
    # load metabric
    np.random.seed(1)
    with open('data/metabric.pkl', 'rb') as f:
        [
            (_, _),
            (_, _),
            (x_val, y_val)
        ] = pickle.load(f)
        t_val = y_val[:, 0]
        y_val = y_val[:, 1]
    # assure that input data same
    t = t_val[:10]
    y = y_val[:10]
    assert np.all(np.equal(y, np.array([1., 1., 1., 0., 1., 0., 1., 0., 1., 1.])))
    assert np.round(np.sum(t), 2) == 1047.37

    # test correct work of DataGenerator
    n_ex_bin = 2
    n_time_bins = 10
    dg = DataGenerator(x=x_val, t=t_val, y=y_val, n_time_bins=n_time_bins, n_ex_bin=n_ex_bin)
    batch_generator = dg.get_batch()

    initial_time_bin_counts = np.array([31, 30, 31, 30, 30, 31, 30, 31, 30, 31])
    assert np.all(np.equal(dg.total_anchors, initial_time_bin_counts))
    ind_to_check = np.array([138, 15, 103, 198, 304, 282, 222, 216, 277, 271, 284, 18, 70, 5, 111, 275, 145, 21])
    x_batch, y_batch, target, all_anchors = next(batch_generator)
    assert np.all(y_val[all_anchors] == y_batch[:, 1])
    assert np.all(t_val[all_anchors] == y_batch[:, 0])

    # test shapes
    assert y_batch.shape == (n_ex_bin * n_time_bins, 2)
    assert x_batch.shape == (n_ex_bin * n_time_bins, 9)

    # test algorithm
    assert np.all(initial_time_bin_counts - dg.total_anchors_cur == n_ex_bin)
    assert np.all(dg.unused_cur[ind_to_check] == 1.)
    x_batch, y_batch, target, _ = next(batch_generator)
    assert np.all(dg.unused_cur[ind_to_check] == 0.)
    assert np.all(initial_time_bin_counts - dg.total_anchors_cur == n_ex_bin * 2)
    assert np.all(np.bincount(target.reshape(target.shape[0])) == n_ex_bin)

    # test comparability matrix
    # calculate on numpy
    comparability_m = get_valid_pairs(t=y_batch[:, 0].reshape(y_batch.shape[0], ),
                                      y=y_batch[:, 1].reshape(y_batch.shape[0], )).astype(int)
    # calculate on tensorflow
    comparability_tf = get_valid_pairs_tf(
        t=y_batch[:, 0].reshape(y_batch.shape[0], ),
        y=y_batch[:, 1].reshape(y_batch.shape[0], )
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        comparability_ = sess.run(comparability_tf)

    # check
    assert np.all(comparability_ == comparability_m)
    assert np.all(np.sum(comparability_, axis=1) == np.array(
        [20, 20, 20, 20,  4, 19,  6, 19, 18, 18, 18, 18, 10, 10, 10, 10, 10, 10, 12, 11], dtype=int))


if __name__ == "__main__":
    test_bg()
