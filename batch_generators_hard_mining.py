import copy

import numpy as np
import tensorflow as tf


def get_time_bins(t, n_time_bins):
    """
    Get equal sized bins
    """
    percent_list = np.linspace(0, 100, n_time_bins + 1, dtype=np.int)
    bins = np.percentile(a=t, q=percent_list[1:-1])
    q = np.digitize(t, bins)
    if n_time_bins != np.unique(q).shape[0]:
        raise Exception("There is too large value for n_time_bins selected")
    return q


def get_valid_pairs(t, y):

    """
    According to definition of Harell's C-index return indices of comparable pairs
    for observation with label y_i and time-to-event t_i
    """
    t_less_i = (t < np.reshape(t, (t.shape[0], 1))).astype(int)
    t_more_i = (t > np.reshape(t, (t.shape[0], 1))).astype(int)
    comp_for_y0 = np.multiply(
        np.reshape(np.ones(y.shape) - y, (y.shape[0], 1)),
        np.multiply(np.reshape(y, (1, y.shape[0])), t_less_i)
    )
    comp_for_y1 = np.multiply(
        np.reshape(y, (y.shape[0], 1)),
        np.multiply(np.reshape(np.ones(y.shape) - y, (1, y.shape[0])), t_more_i) + y
    )
    comparability_m = comp_for_y1 + comp_for_y0
    return comparability_m


def get_valid_pairs_tf(t, y):

    """
    According to definition of Harell's C-index return indices of comparable pairs
    for observation with label y_i and time-to-event t_i
    """
    y = tf.cast(y, tf.int8)
    t_less_i = tf.cast(t < tf.reshape(t, (tf.shape(t)[0], 1)), tf.int8)
    t_more_i = tf.cast(t > tf.reshape(t, (tf.shape(t)[0], 1)), tf.int8)

    comp_for_y0 = tf.multiply(
        tf.reshape(tf.ones(tf.shape(y), dtype=tf.int8) - y, (tf.shape(y)[0], 1)),
        tf.multiply(tf.reshape(y, (1, tf.shape(y)[0])), t_less_i)
    )
    comp_for_y1 = tf.multiply(
        tf.reshape(y, (tf.shape(y)[0], 1)),
        tf.multiply(tf.reshape(tf.ones(tf.shape(y), dtype=tf.int8) - y, (1, tf.shape(y)[0])), t_more_i) + y
    )
    comparability_m = comp_for_y1 + comp_for_y0
    return comparability_m


class DataGenerator(object):
    """" Generates pairs
    The target is defined for pairs comparable in terms of Harell's C-index
    The samples in a batch are taken uniformly from event time distribution
    DataGenerator for online batch generation
    """
    def __init__(self, x, t, y, n_time_bins, n_ex_bin):
        """
        Each batch will consist of pairs generated for n_ex_bin examples from each of n_time_bins time bins
        """

        self.x = x
        self.t = t
        self.y = y
        self.n_time_bins = n_time_bins
        self.n_ex_bin = n_ex_bin
        self.unused_cur = None
        self.total_anchors_cur = None

        # split examples on given number of time bins and get time bin index for each example
        self.q = get_time_bins(t=self.t, n_time_bins=self.n_time_bins)
        self.ind_sorted_by_t = np.argsort(self.t)
        self.total_anchors = np.bincount(self.q)
        self.bins_borders = np.cumsum(self.total_anchors)

    def get_batch(self):

        self.unused_cur = np.ones(self.t.shape)
        self.total_anchors_cur = copy.deepcopy(self.total_anchors)
        # get next batch if there are enough examples in each time bin
        while np.all(self.total_anchors_cur > self.n_ex_bin):
            yield self.batch_generator()
        return None

    def batch_generator(self):
        """
        Returns:
            x_batch - features of batch examples
            y_batch - [duration(t) of batch examples, duration(t) of batch examples, event label(y) of batch examples,
                event label(y) of batch examples]
            time_bin - array of time bin for each batch example
        """
        # indices of examples in a new batch
        all_anchors = np.array([], dtype=int)
        bin_right_border_cur = 0
        for bin_right_border in self.bins_borders:
            ind_in_bin = self.ind_sorted_by_t[bin_right_border_cur: (bin_right_border - 1)]
            ind_in_bin_used = self.unused_cur[ind_in_bin]
            ind_proba = ind_in_bin_used / np.sum(ind_in_bin_used)
            cur_anc = np.random.choice(a=ind_in_bin, size=self.n_ex_bin, replace=False, p=ind_proba)
            bin_right_border_cur = bin_right_border
            all_anchors = np.append(all_anchors, cur_anc)
        # set to zero to not sample these examples later
        self.unused_cur[all_anchors] = 0
        # recalculate number of examples in each bin
        self.total_anchors_cur = np.bincount(self.q[self.unused_cur == 1])

        # fill in matrices with features
        x_batch = self.x[all_anchors, :]
        y_batch = np.hstack([
            self.t[all_anchors].reshape(all_anchors.shape[0], 1),
            self.y[all_anchors].reshape(all_anchors.shape[0], 1),
        ])
        # save time bin of example
        time_bin = np.reshape(self.q[all_anchors], (all_anchors.shape[0], 1))
        return x_batch, y_batch, time_bin, all_anchors
