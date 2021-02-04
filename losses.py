import tensorflow as tf

from batch_generators_hard_mining import get_valid_pairs_tf


def calc_survival_value(alphas, betas, t):
    s = tf.exp(-1 * tf.pow(tf.divide(t, alphas + 1e-6), betas))
    return s


def calc_likelyhood(alphas, betas, t, y):
    ev = tf.multiply(y, tf.add(tf.multiply(betas, tf.math.log(tf.divide(t, alphas + 1e-6))), tf.math.log(betas)))
    lh = tf.subtract(ev, tf.pow(tf.divide(t, alphas + 1e-6), betas))
    return lh


def binary_cross_entropy_loss(t_a, t_b, alphas_a, betas_a, alphas_b, betas_b, target, sample_weight):
    s_a = calc_survival_value(t=t_a, alphas=alphas_a, betas=betas_a)
    s_b = calc_survival_value(t=t_b, alphas=alphas_b, betas=betas_b)
    sigm = tf.nn.sigmoid(s_a - s_b)
    sigm = tf.clip_by_value(sigm, 1e-6, 1 - 1e-6)
    label_pos = tf.multiply(target, tf.log(sigm + 1e-6))
    label_neg = tf.multiply(1 - target, tf.multiply(tf.log(1 + 1e-6 - sigm), sample_weight))
    ll = tf.add(label_pos, label_neg)
    mean_ll = -1 * tf.reduce_mean(ll)
    return mean_ll


def weibull_loglikelyhood_loss(t, y, alphas, betas):
    lh_a = calc_likelyhood(t=t, y=y, alphas=alphas, betas=betas)
    mean_lh = -1 * tf.reduce_mean(lh_a)
    return mean_lh


def contrastive_margin_loss(output_tr_a, output_tr_b, target, sample_weight):
    dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(output_tr_a, output_tr_b), axis=1) + 1e-32)
    label_pos = tf.multiply(target, dist)
    label_neg = tf.multiply(1 - target, tf.math.maximum(sample_weight - dist, 0))
    ll = tf.add(label_pos, label_neg)
    mean_loss = tf.reduce_mean(ll)
    return mean_loss


def get_contrastive_positive_label(batch_time_bin):
    # check same time bin
    batch_time_bin_reshaped = tf.reshape(batch_time_bin, (tf.shape(batch_time_bin)[0], ))
    pos_label = tf.cast(tf.equal(batch_time_bin_reshaped, batch_time_bin), dtype=tf.int8)
    # check that i and j are different examples
    indices_equal = tf.eye(tf.shape(batch_time_bin)[0], dtype=tf.int8)
    indices_not_equal = 1 - indices_equal
    mask = tf.multiply(indices_not_equal, pos_label)
    return mask


def get_contrastive_negative_label(batch_time_bin):
    # check same time bin
    batch_time_bin_reshaped = tf.reshape(batch_time_bin, (tf.shape(batch_time_bin)[0], ))
    label = 1 - tf.cast(tf.equal(batch_time_bin_reshaped, batch_time_bin), dtype=tf.int8)
    return label


def calc_batch_distances(v):
    dot_product = tf.matmul(v, tf.transpose(v))
    # L2 norm of each embedding is on diagonal of dot_product
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)
    # to achieve numerical stability of sqrt
    mask = tf.to_float(tf.equal(distances, 0.0))
    distances = distances + mask * 1e-16
    distances = tf.sqrt(distances)
    distances = distances * (1.0 - mask)
    return distances


def get_delta_time_sample_weight(time_bin):
    delta_time_bin = tf.subtract(tf.reshape(time_bin, (tf.shape(time_bin)[0], )), time_bin)
    n_time_bins = tf.add(
        tf.subtract(tf.reduce_max(time_bin), tf.reduce_min(time_bin)),
        tf.constant(1, dtype=time_bin.dtype)
    )
    sample_weight = tf.add(tf.constant(1, dtype=tf.float32), tf.cast(tf.abs(delta_time_bin) / n_time_bins, tf.float32))
    return delta_time_bin, sample_weight


def batch_hard_sampling_contrastive_loss(output_tr, time_bin, t, y, margin_scale):
    # get labels
    pos_label = get_contrastive_positive_label(time_bin)
    neg_label = get_contrastive_negative_label(time_bin)
    # assert pairs are comparable in terms of concordance
    comparability_m = get_valid_pairs_tf(t, y)
    pos_label = tf.to_float(tf.multiply(comparability_m, pos_label))
    neg_label = tf.to_float(tf.multiply(comparability_m, neg_label))
    # calc distance between representations
    distances = calc_batch_distances(output_tr)

    # get hardest positive examples
    anchor_positive_dist = tf.multiply(pos_label, distances)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=False)
    # filter relevant positive examples
    mask_greater_zero = tf.greater(hardest_positive_dist, tf.constant(0, dtype=tf.float32))
    hardest_positive_dist = tf.boolean_mask(hardest_positive_dist,  mask_greater_zero)
    hardest_positive_dist = tf.reshape(hardest_positive_dist, (tf.shape(hardest_positive_dist)[0], 1))

    # get hardest negative examples
    # bias (increase) the distances for positive examples
    max_anchor_dist = tf.reduce_max(distances, axis=1, keepdims=True)
    anchor_negative_dist = tf.add(distances,
                                  max_anchor_dist * (tf.subtract(tf.constant(1, dtype=tf.float32), neg_label))
                                  )
    hardest_negative_dist = tf.cast(tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True), tf.float32)
    ind_min = tf.argmin(anchor_negative_dist, axis=1)
    ind_min = tf.expand_dims(ind_min, 1)
    range_rows = tf.cast(tf.expand_dims(tf.range(tf.shape(ind_min)[0]), 1), tf.int64)
    ind_min = tf.concat([range_rows, ind_min], axis=1)
    # introduce margin
    _, sample_weight = get_delta_time_sample_weight(time_bin)
    sample_weight = margin_scale * sample_weight
    sample_weight = tf.gather_nd(sample_weight, ind_min)
    hardest_negative_dist_margin = tf.math.maximum(
        tf.subtract(tf.reshape(sample_weight, (tf.shape(sample_weight)[0], 1)), hardest_negative_dist),
        tf.constant(0, dtype=tf.float32)
    )
    # filter relevant negative examples
    n_possible_pairs = tf.reduce_sum(neg_label, axis=1)
    mask_exist_possible_pairs = tf.greater(n_possible_pairs, tf.constant(0, dtype=tf.float32))
    mask_exist_possible_pairs = tf.reshape(mask_exist_possible_pairs, (tf.shape(mask_exist_possible_pairs)[0], 1))
    mask_loss_greater_zero = tf.greater(hardest_negative_dist_margin, tf.constant(0, dtype=tf.float32))
    mask_negative = tf.logical_and(mask_exist_possible_pairs, mask_loss_greater_zero)
    hardest_negative_dist_margin = tf.boolean_mask(hardest_negative_dist_margin, mask_negative)
    hardest_negative_dist_margin = tf.reshape(hardest_negative_dist_margin,
                                              (tf.shape(hardest_negative_dist_margin)[0], 1)
                                              )
    # final loss
    loss = tf.concat([hardest_positive_dist, hardest_negative_dist_margin], axis=0)
    mean_loss = tf.reduce_mean(loss)
    return hardest_positive_dist, hardest_negative_dist_margin, mean_loss


def batch_hard_sampling_cross_entropy_loss(output_tr, time_bin, t, y):
    sh = tf.shape(t)
    alphas = tf.reshape(output_tr[:, 0], sh)
    betas = tf.reshape(output_tr[:, 1], sh)
    # get weight for each pair based on difference in time bins
    delta_time_bin, sample_weight = get_delta_time_sample_weight(time_bin)
    # assert pairs are comparable in terms of concordance
    comparability_m = get_valid_pairs_tf(t, y)
    # pos_label: 1 if delta_time_bin > 0
    pos_label = get_cross_entropy_positive_label(delta_time_bin)
    # neg_label: 1 if delta_time_bin < 0
    neg_label = get_cross_entropy_negative_label(delta_time_bin)
    pos_label = tf.to_float(tf.multiply(comparability_m, pos_label))
    neg_label = tf.to_float(tf.multiply(comparability_m, neg_label))
    # get survival value prediction
    surv = tf.cast(calc_survival_value(alphas, betas, t), dtype=tf.float32)
    delta_surv = tf.subtract(tf.reshape(surv, (1, tf.shape(surv)[0])), surv)

    # hardest positive examples
    hardest_positive = tf.multiply(pos_label, tf.multiply(delta_surv, sample_weight))
    mask_less_zero = tf.cast(tf.less(hardest_positive, tf.constant(0, dtype=hardest_positive.dtype)),
                             dtype=hardest_positive.dtype)
    hardest_positive = tf.multiply(hardest_positive, mask_less_zero)
    hardest_positive = tf.reduce_min(hardest_positive, axis=1, keepdims=False)
    # proba
    hardest_pos_sigm = tf.nn.sigmoid(hardest_positive)
    hardest_pos_sigm = tf.clip_by_value(hardest_pos_sigm, 1e-6, 1 - 1e-6)
    # log loss
    loss_pos = tf.log(hardest_pos_sigm + 1e-6)

    # hardest negative examples
    hardest_negative = tf.multiply(neg_label, tf.multiply(delta_surv, sample_weight))
    mask_greater_zero = tf.cast(tf.greater(hardest_negative, tf.constant(0, dtype=hardest_negative.dtype)),
                                dtype=hardest_negative.dtype)
    hardest_negative = tf.multiply(hardest_negative, mask_greater_zero)
    hardest_negative = tf.reduce_max(hardest_negative, axis=1, keepdims=False)
    # proba
    hardest_neg_sigm = tf.nn.sigmoid(hardest_negative)
    hardest_neg_sigm = tf.clip_by_value(hardest_neg_sigm, 1e-6, 1 - 1e-6)
    # log loss
    loss_neg = tf.log(1 + 1e-6 - hardest_neg_sigm)
    # final loss
    loss = tf.concat([loss_neg, loss_pos], axis=0)
    mean_loss = -1 * tf.reduce_mean(loss)
    return hardest_pos_sigm, hardest_neg_sigm, mean_loss


def get_cross_entropy_positive_label(delta_time_bin):
    # for comparable in terms of concordance pairs the greater the time, the higher the survival
    pos_label = tf.cast(tf.greater(delta_time_bin, tf.constant(0, dtype=delta_time_bin.dtype)), dtype=tf.int8)
    return pos_label


def get_cross_entropy_negative_label(delta_time_bin):
    # for comparable in terms of concordance pairs the lower the time, the lower the survival
    neg_label = tf.cast(tf.less(delta_time_bin, tf.constant(0, dtype=delta_time_bin.dtype)), dtype=tf.int8)
    return neg_label
