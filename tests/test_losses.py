import pickle

import numpy as np
import tensorflow as tf

from batch_generators_hard_mining import get_valid_pairs_tf, ContrastiveDataGenerator
from losses import get_contrastive_positive_label, calc_batch_distances, batch_hard_sampling_contrastive_loss, \
    get_contrastive_negative_label


def test_losses():
    np.random.seed(1)
    # load metabric
    with open('../data/metabric.pkl', 'rb') as f:
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

    # get batch generator
    n_ex_bin = 3
    n_time_bins = 10
    dg = ContrastiveDataGenerator(x=x_val, t=t_val, y=y_val, n_time_bins=n_time_bins, n_ex_bin=n_ex_bin)
    batch_generator = dg.get_batch()
    [x_batch_left, x_batch_right], y_batch, sample_weight, target = next(batch_generator)

    # calculate comparability
    comparability_tf = get_valid_pairs_tf(
        t=y_batch[:, 0].reshape(y_batch.shape[0], ),
        y=y_batch[:, 2].reshape(y_batch.shape[0], )
    )
    positive_mask = get_contrastive_positive_label(target)
    negative_mask = get_contrastive_negative_label(target)
    dist = calc_batch_distances(x_batch_left)
    hardest_positive_dist, hardest_negative_dist, mean_loss = batch_hard_sampling_contrastive_loss(
        output_tr=x_batch_left, time_bin=target,
        t=y_batch[:, 0].reshape(y_batch.shape[0], ),
        y=y_batch[:, 2].reshape(y_batch.shape[0], )
    )
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        comparability_ = sess.run(comparability_tf)
        assert np.sum(comparability_) == 577
        pos_label_mask = sess.run(positive_mask)
        assert np.all(np.sum(pos_label_mask, axis=1) == n_ex_bin - 1)

        pos_label_mask = sess.run(tf.multiply(pos_label_mask, comparability_))
        assert np.sum(pos_label_mask, axis=1)[-1] == 0

        neg_label_mask = sess.run(negative_mask)
        neg_label_mask = sess.run(tf.multiply(neg_label_mask, comparability_))
        assert np.sum(neg_label_mask, axis=1)[0] == 0

        distances = sess.run(dist)
        assert distances[0, 0] == 0

        # test number of total positive examples
        positive_dist = sess.run(hardest_positive_dist)
        assert positive_dist.shape == (15, 1)
        # test number of total negative examples
        negative_dist = sess.run(hardest_negative_dist)
        assert negative_dist.shape == (9, 1)

        loss = sess.run(mean_loss)
        assert round(loss) == 2


if __name__ == "__main__":
    test_losses()
