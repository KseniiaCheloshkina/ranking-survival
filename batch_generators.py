import numpy as np
import copy


class DataGenerator(object):

    @staticmethod
    def get_relevant_contragents(t, y, q):

        """
        According to definition of Harell's C-index return indices of comparable pairs
        for observation with label y_i and time-to-event t_i
        """
        
        n = t.shape[0]
        comparability_m = np.zeros((n, n))
        target_m = np.zeros((n, n))
        dq_m = np.zeros((n, n))
        for i in range(n):
            if y[i] == 0:
                comparability_m[i, :] = np.multiply(y, (t < t[i]).astype(int))
                target_m[i, :] = comparability_m[i, :]
            else:
                comparability_m[i, :] = np.multiply(1 - y, (t > t[i]).astype(int)) + y
                # target_m[i, :] =  np.multiply(y, (t > t[i]).astype(int)) # 0
                target_m[i, :] = np.multiply(y, (t < t[i]).astype(int))   # 1
                # target_m[i, :] =  np.multiply(1 - y, (t > t[i]).astype(int)) # 0
            dq_m[i, :] = q - q[i]

        return comparability_m, dq_m, target_m

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

    def __init__(self, x, t, y, batch_size, method="binary", **kwargs):
        
        self.x = x
        self.t = t
        self.y = y
        self.half_batch_size = int(batch_size // 2)
        self.batch_size = self.half_batch_size * 2
        self.ij_neg_sorted_cur = None
        self.ij_pos_sorted_cur = None
        
        kw = kwargs.keys()
        
        if method == "binary":
            self.batch_generator = self.binary_cross_entropy_generator
            if 'n_time_bins' not in kw:
                raise Exception("It is needed to specify argument n_time_bins for method 'binary'")
            else:
                self.n_time_bins = kwargs['n_time_bins']

            # get needed time-event stats
            self.q = self.get_time_bins(t=self.t, n_time_bins=self.n_time_bins)        
            self.comparability_m, self.dq_m, self.target_m = self.get_relevant_contragents(self.t, self.y, self.q)
            
            # define all possible examples
            self.pos_ex = np.multiply(self.comparability_m, self.target_m)
            self.neg_ex = np.multiply(self.comparability_m, 1 - self.target_m)

            # indices of positive pairs
            self.ij_pos = np.where(self.pos_ex == 1)
            self.ij_pos = np.vstack(self.ij_pos)
            # indices of negative pairs
            self.ij_neg = np.where(self.neg_ex == 1)
            self.ij_neg = np.vstack(self.ij_neg)

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

    def get_batch(self):
        
        self.ij_neg_sorted_cur = copy.deepcopy(self.ij_neg_sorted)
        self.ij_pos_sorted_cur = copy.deepcopy(self.ij_pos_sorted)

        while (self.ij_neg_sorted_cur.shape[1] >= self.half_batch_size) & \
                (self.ij_pos_sorted_cur.shape[1] >= self.half_batch_size):
            yield self.batch_generator()
        return None

    def binary_cross_entropy_generator(self):

        """
        Keyword arguments:
        """
        
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
                dq_cur[i] = self.dq_m[ex[0]][ex[1]]

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
