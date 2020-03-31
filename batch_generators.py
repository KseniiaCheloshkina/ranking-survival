import numpy as np

def binary_cross_entropy_batch_generator(x, t, y, n_time_bins=10, batch_anchors=1, anchor_pairs=2):
    
    # TODO: баланс классов (убедиться, что будет генерироваться и положительные и отрицательные примеры, 
    # с одинаковой вероятностью берем бины справа и слева от бина исходного примера)
    # TODO: предрасчитать то, что можно не пересчитывать каждый раз  
    
    """
    Keyword arguments:
    sample_weight - is it needed to weight examples proportionally to time bin during sampling and training
     # sample_weight - q для train_on_batch
    """
    assert anchor_pairs > 1
    x_batch_left = np.zeros((batch_anchors * anchor_pairs, x.shape[1]))
    x_batch_right = np.zeros((batch_anchors * anchor_pairs, x.shape[1]))
    y_batch = np.zeros((batch_anchors * anchor_pairs, 4))
    sample_weight = np.zeros((batch_anchors * anchor_pairs, 1))
    q = get_time_bins(t=t, n_time_bins=n_time_bins)
    
    # uniform sampling of time bins
    bins = np.random.uniform(0, n_time_bins - 1, batch_anchors).astype(int)
    # for each anchor
    i = 0
    for b in bins:
        ind_anchor = np.random.choice(np.where(q == b)[0], 1, replace=False)[0]
        x_anchor = x[ind_anchor, :]
        t_anchor = t[ind_anchor]
        y_anchor = y[ind_anchor]
        x_batch_left[(anchor_pairs * i):(anchor_pairs * i + anchor_pairs), :] = x_anchor
        y_batch[(anchor_pairs * i):(anchor_pairs * i + anchor_pairs), 0] = t_anchor
        # find possible pair
        ind_pos_contr = filter_relevant_contragent(t, y, t_anchor, y_anchor)
        dq = q[ind_pos_contr] - q[ind_anchor] 
        # half of times from the same bin and the other half - uniformly from all bins
        n_obs_same_bin = dq[dq == 0].shape[0]
        n_obs_same_bin = min(n_obs_same_bin, anchor_pairs // 2)
        if n_obs_same_bin == 0:
            pairs_bins = np.random.uniform(np.min(dq), np.max(dq), anchor_pairs).astype(int)
        else:
            pairs_bins = np.repeat(0, repeats=n_obs_same_bin)
            pairs_bins = np.append(pairs_bins, np.random.uniform(np.min(dq), np.max(dq), anchor_pairs - n_obs_same_bin).astype(int))
        # TODO: proportional sampling
        # for each pair
        j = 0        
        for p_b in pairs_bins:
            ind_pair = np.random.choice(np.where(dq == p_b)[0], 1, replace=False)[0]
            x_batch_right[i * anchor_pairs + j, :] = x[ind_pos_contr, :][ind_pair]
            y_batch[i * anchor_pairs + j, 1] = t[ind_pos_contr][ind_pair]
            y_batch[i * anchor_pairs + j, 2] = np.int(dq[ind_pair] == 0)
            y_batch[i * anchor_pairs + j, 3] = 1 + abs(dq[ind_pair]) / n_time_bins
            j += 1
        i += 1

    yield [x_batch_left, x_batch_right], y_batch
