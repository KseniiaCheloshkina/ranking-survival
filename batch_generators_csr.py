import numpy as np
import copy
from scipy.sparse import csr_matrix
import tqdm
import time


class CommonDataGenerator(object):
    """" Generates pairs
    The target is defined for pairs comparable in terms of Harell's C-index
    The samples in a batch are taken uniformly from event time distribution
    """

    @staticmethod
    def get_time_bins(t, n_time_bins):
        """
        Get equal size bins
        """
        percent_list = np.linspace(0, 100, n_time_bins + 1, dtype=np.int)
        bins = np.percentile(a=t, q=percent_list[1:-1])
        q = np.digitize(t, bins)
        if n_time_bins != np.unique(q).shape[0]:
            raise Exception("There is too large value for n_time_bins selected")
        return q

    @staticmethod
    def get_relevant_contragents(t, y, q, pairs_per_sample):

        """
        According to definition of Harell's C-index return indices of comparable pairs
        for observation with label y_i and time-to-event t_i
        """

        n = t.shape[0]
        m = np.max(t)

        y_csr = csr_matrix(y)
        y_csr = y_csr.transpose()
        y_nonzero_ind = y_csr.nonzero()[0]
        y_zero_ind = np.setdiff1d(np.arange(0, y.shape[0]), y_nonzero_ind)
        comparability_m = csr_matrix((n, n))
        target_m = csr_matrix((n, n))
        dq_m = csr_matrix((n, n))
        dq_m_0 = csr_matrix((n, n))

        t_uniq, uniq_indices = np.unique(t, return_inverse=True)

        # we will overwrite this matrix with each t_cur (on each step contains indices of elements for which t < t_cur)
        t_less_ti = csr_matrix((n, 1))
        # we will overwrite this matrix with each t_cur (on each step contains indices of elements for which t > t_cur)
        t_more_ti = csr_matrix(np.ones((n, 1)))
        ind_t_un = np.where(t_uniq == 1)[0]
        ind_t_cur = np.where(uniq_indices == ind_t_un)[0]

        # we will iterate through observations with t == t_cur
        for t_cur in tqdm.notebook.tqdm(range(2, m)):
            # keep only those which are less than t_cur - 1
            t_less_ti += csr_matrix(
                (
                    np.ones(ind_t_cur.shape[0]),
                    (
                        ind_t_cur,
                        np.repeat(0, ind_t_cur.shape[0])
                    )
                ),
                shape=(n, 1)
            )
            # keep only those which are greater than t_cur - 1
            t_more_ti -= csr_matrix(
                (
                    np.ones(ind_t_cur.shape[0]),
                    (
                        ind_t_cur,
                        np.repeat(0, ind_t_cur.shape[0])
                    )
                ),
                shape=(n, 1)
            )

            ind_t_un = np.where(t_uniq == t_cur)[0]
            ind_t_cur = np.where(uniq_indices == ind_t_un)[0]

            t_cur_y_0 = np.intersect1d(y_zero_ind, ind_t_cur)
            # find comparable examples for observations with t == t_cur and y = 0
            res_0 = y_csr.multiply(t_less_ti)
            # t_cur_y_0 are comparable with res[res == 1]
            m2 = t_cur_y_0.shape[0]
            if res_0.count_nonzero() >= m2 * pairs_per_sample:
                # sample pair for each example
                ind_nonzero_sampled = np.random.choice(res_0.nonzero()[0], size=m2 * pairs_per_sample, replace=False)
                final_comp_pairs = csr_matrix(
                    (
                        np.ones((pairs_per_sample * m2,)),
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))
                comparability_m += final_comp_pairs
                target_m += final_comp_pairs
                dq_m += csr_matrix(
                    (
                        q[ind_nonzero_sampled] - q[np.repeat(t_cur_y_0, pairs_per_sample)],
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))
                dq_m_0 += csr_matrix(
                    (
                        (q[ind_nonzero_sampled] == q[np.repeat(t_cur_y_0, pairs_per_sample)]).astype(int),
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))

            t_cur_y_0 = np.intersect1d(y_nonzero_ind, ind_t_cur)
            # find comparable examples for observations with t == t_cur and y = 1
            res_1 = t_more_ti.multiply(csr_matrix(np.ones((n, 1))) - y_csr) + y_csr
            # t_cur_y_0 are comparable with res[res == 1]
            m2 = t_cur_y_0.shape[0]
            if res_1.count_nonzero() >= m2 * pairs_per_sample:
                # sample pair for each example
                ind_nonzero_sampled = np.random.choice(res_1.nonzero()[0], size=m2 * pairs_per_sample, replace=False)
                comparability_m += csr_matrix(
                    (
                        np.ones((pairs_per_sample * m2,)),
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))
                target_m += csr_matrix(
                    (
                        np.reshape(res_0[ind_nonzero_sampled, :].toarray(), ind_nonzero_sampled.shape),
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))
                dq_m += csr_matrix(
                    (
                        q[ind_nonzero_sampled] - q[np.repeat(t_cur_y_0, pairs_per_sample)],
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))
                dq_m_0 += csr_matrix(
                    (
                        (q[ind_nonzero_sampled] == q[np.repeat(t_cur_y_0, pairs_per_sample)]).astype(int),
                        (
                            np.repeat(t_cur_y_0, pairs_per_sample),
                            ind_nonzero_sampled
                        )
                    ),
                    shape=(n, n))

        return comparability_m, target_m, dq_m, dq_m_0

    def __init__(self, x, t, y, batch_size, n_time_bins, pairs_per_sample):

        self.x = x
        self.t = t
        self.y = y
        self.n_time_bins = n_time_bins
        self.half_batch_size = int(batch_size // 2)
        self.batch_size = self.half_batch_size * 2
        self.ij_neg_sorted = None
        self.ij_pos_sorted = None
        self.ij_neg_sorted_cur = None
        self.ij_pos_sorted_cur = None
        self.ij_pos = None
        self.ij_neg = None
        self.pos_ex = None
        self.neg_ex = None
        self.constant_target = None

        # get needed time-event stats
        self.q = self.get_time_bins(t=self.t, n_time_bins=self.n_time_bins)
        self.comparability_m, self.target_m, self.dq_m, self.dq_m_0 = self.get_relevant_contragents(
            self.t, self.y, self.q, pairs_per_sample)

    def get_stats(self):
        # indices of positive pairs
        cols_ind = self.pos_ex.indices
        rows_ind = np.array([], dtype='int')
        for i in range(1, self.pos_ex.indptr.shape[0]):
            rows_ind = np.append(rows_ind, np.repeat(a=i - 1,
                                                     repeats=self.pos_ex.indptr[i] - self.pos_ex.indptr[i - 1]))
        self.ij_pos = np.vstack([rows_ind, cols_ind])
        # indices of negative pairs
        cols_ind = self.neg_ex.indices
        rows_ind = np.array([], dtype='int')
        for i in range(1, self.neg_ex.indptr.shape[0]):
            rows_ind = np.append(rows_ind, np.repeat(a=i - 1,
                                                     repeats=self.neg_ex.indptr[i] - self.neg_ex.indptr[i - 1]))
        self.ij_neg = np.vstack([rows_ind, cols_ind])
        # sort examples by bin to sample uniformly over bins (sort by first element bin)
        pos_ind_sorted_by_q = np.argsort(self.q[self.ij_pos[0]])
        self.ij_pos_sorted = self.ij_pos[:, pos_ind_sorted_by_q]
        neg_ind_sorted_by_q = np.argsort(self.q[self.ij_neg[0]])
        self.ij_neg_sorted = self.ij_neg[:, neg_ind_sorted_by_q]
        self.constant_target = np.hstack(
            [
                np.repeat(a=1, repeats=self.half_batch_size),
                np.repeat(a=0, repeats=self.half_batch_size)
            ]
        )
        return self

    def get_batch(self):

        self.ij_neg_sorted_cur = copy.deepcopy(self.ij_neg_sorted)
        self.ij_pos_sorted_cur = copy.deepcopy(self.ij_pos_sorted)

        while (self.ij_neg_sorted_cur.shape[1] >= self.half_batch_size) & \
                (self.ij_pos_sorted_cur.shape[1] >= self.half_batch_size):
            yield self.batch_generator()
        return None

    def batch_generator(self):

        x_batch_left = np.zeros((self.batch_size, self.x.shape[1]))
        x_batch_right = np.zeros((self.batch_size, self.x.shape[1]))
        y_batch = np.zeros((self.batch_size, 4))
        sample_weight = np.zeros((self.batch_size, 1))

        def sample_examples(ij, start_ind, end_ind):
            # sample
            cur_anc = np.random.choice(a=ij.shape[1], size=self.half_batch_size, replace=False)
            p = ij[:, cur_anc]
            dq_cur = np.zeros((self.half_batch_size,))
            for i in range(p.shape[1]):
                ex = p[:, i]
                dq_cur[i] = self.dq_m[ex[0], ex[1]]

            # fill batch data with positive examples
            x_batch_left[start_ind:end_ind, :] = self.x[p[0], :]
            x_batch_right[start_ind:end_ind, :] = self.x[p[1], :]
            y_batch[start_ind:end_ind, 0] = self.t[p[0]]
            y_batch[start_ind:end_ind, 1] = self.t[p[1]]
            y_batch[start_ind:end_ind, 2] = self.y[p[0]]
            y_batch[start_ind:end_ind, 3] = self.y[p[1]]
            sample_weight[start_ind:end_ind, 0] = 1 + abs(dq_cur) / self.n_time_bins
            return x_batch_left, x_batch_right, y_batch, sample_weight, cur_anc

        # sample positive examples
        x_batch_left, x_batch_right, y_batch, sample_weight, used_ind = sample_examples(
            ij=self.ij_pos_sorted_cur,
            start_ind=0,
            end_ind=self.half_batch_size)

        # remove used positive examples
        self.ij_pos_sorted_cur = np.delete(self.ij_pos_sorted_cur, used_ind, 1)

        # sample negative examples
        x_batch_left, x_batch_right, y_batch, sample_weight, used_ind = sample_examples(
            ij=self.ij_neg_sorted_cur,
            start_ind=self.half_batch_size,
            end_ind=self.batch_size)
        # remove used positive examples
        self.ij_neg_sorted_cur = np.delete(self.ij_neg_sorted_cur, used_ind, 1)

        return [x_batch_left, x_batch_right], y_batch, sample_weight, self.constant_target


class ContrastiveDataGenerator(CommonDataGenerator):
    """" Generates pairs for contrastive loss
    Positive pairs consist of observations from the same time bin
    Negative pairs consist of observations from different time bins
    Each batch has balanced number of positive and negative examples.
    The target is defined for pairs comparable in terms of Harell's C-index
    The samples in a batch are taken uniformly from event time distribution
    Note: number of positive examples is significantly lower than number
    of negative examples
    """
    def __init__(self, x, t, y, batch_size, n_time_bins, pairs_per_sample):
        print("Initialization of batch generator...")
        super().__init__(x, t, y, batch_size, n_time_bins, pairs_per_sample)
        print("define all positive examples")
        # define all possible and all negative examples
        self.pos_ex = self.comparability_m.multiply(self.dq_m_0)
        print("define all negative examples")
        self.neg_ex = self.comparability_m - self.pos_ex
        print("get stats")
        self.get_stats()
        print("Initialization of batch generator is completed")


class BinaryDataGenerator(CommonDataGenerator):
    """" Generates pairs for contrastive loss
    Positive pairs consist of observations from the same time bin
    Negative pairs consist of observations from different time bins
    Each batch has balanced number of positive and negative examples.
    The target is defined for pairs comparable in terms of Harell's C-index
    The samples in a batch are taken uniformly from event time distribution
    """
    def __init__(self, x, t, y, batch_size, n_time_bins, pairs_per_sample):
        print(time.localtime())
        print("Initialization of batch generator...")
        super().__init__(x, t, y, batch_size, n_time_bins, pairs_per_sample)
        print(time.localtime())
        print("define all positive examples")
        # define all possible and all negative examples
        self.pos_ex = self.comparability_m.multiply(self.target_m)
        print(time.localtime())
        print("define all negative examples")
        self.neg_ex = self.comparability_m - self.pos_ex
        print(time.localtime())
        print("get stats")
        self.get_stats()
        print(time.localtime())
        print("Initialization of batch generator is completed")
