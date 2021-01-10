import pickle
import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), "data"))
sys.path.append(os.getcwd())

from batch_generators_hard_mining import get_valid_pairs_tf, DataGenerator
from losses import get_contrastive_positive_label, calc_batch_distances, batch_hard_sampling_contrastive_loss, \
    get_contrastive_negative_label, batch_hard_sampling_cross_entropy_loss, get_delta_time_sample_weight


def test_losses_hard_mining():
    np.random.seed(1)
    # load metabric
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

    # get batch generator
    n_ex_bin = 3
    n_time_bins = 10
    dg = DataGenerator(x=x_val, t=t_val, y=y_val, n_time_bins=n_time_bins, n_ex_bin=n_ex_bin)
    batch_generator = dg.get_batch()
    x_batch, y_batch, target, _ = next(batch_generator)

    # calculate comparability
    comparability_tf = get_valid_pairs_tf(
        t=y_batch[:, 0].reshape(y_batch.shape[0], ),
        y=y_batch[:, 1].reshape(y_batch.shape[0], )
    )
    # test functions for contrastive loss
    positive_mask = get_contrastive_positive_label(target)
    negative_mask = get_contrastive_negative_label(target)
    dist = calc_batch_distances(x_batch)
    hardest_positive_dist, hardest_negative_dist, mean_loss = batch_hard_sampling_contrastive_loss(
        output_tr=x_batch, time_bin=target,
        t=y_batch[:, 0].reshape(y_batch.shape[0], ),
        y=y_batch[:, 1].reshape(y_batch.shape[0], ),
        margin_scale=1
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
        assert positive_dist.shape == (24, 1)
        # test number of total negative examples
        negative_dist = sess.run(hardest_negative_dist)
        assert negative_dist.shape == (9, 1)
        loss = sess.run(mean_loss)
        assert round(loss) == 2

    # test functions for cross entropy loss
    delta_time_bin, sample_weight = get_delta_time_sample_weight(target)

    hardest_pos_sigm, hardest_neg_sigm, mean_loss = batch_hard_sampling_cross_entropy_loss(
        output_tr=tf.maximum(x_batch, 1e-3), time_bin=target,
        t=y_batch[:, 0].reshape(y_batch.shape[0], 1),
        y=y_batch[:, 1].reshape(y_batch.shape[0], 1)
    )

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        assert sess.run(tf.reduce_sum(tf.linalg.diag_part(delta_time_bin, name='diag_part'))) == 0
        hardest_pos_sigm_ = sess.run(hardest_pos_sigm)
        assert np.all(hardest_pos_sigm_ <= 0.5)
        hardest_neg_sigm_ = sess.run(hardest_neg_sigm)
        assert np.all(hardest_neg_sigm_ >= 0.5)
        loss_ = sess.run(mean_loss)
        assert round(loss_ * 100) == 79.0


if __name__ == "__main__":
    test_losses_hard_mining()
