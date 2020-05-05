import tensorflow as tf


def calc_survival_value(alphas, betas, t):
    s = tf.exp(-1 * tf.pow(tf.divide(t, alphas + 1e-6), betas))
    return s


def calc_likelyhood(alphas, betas, t, y):
    ev = tf.multiply(y, tf.add(tf.multiply(betas, tf.log(tf.divide(t, alphas + 1e-6))), tf.log(betas)))
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
