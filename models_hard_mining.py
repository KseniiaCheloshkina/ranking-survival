import random

import numpy as np
import tensorflow as tf

import losses
from models import metabric_main_network


class WeibullModel(object):

    def __init__(self, input_shape, main_network=metabric_main_network, seed=7, alpha_reg=1e-3,
                 alpha_bias_random_mean=0.0, alpha_random_stddev=1.0, beta_random_stddev=1.0):
        self.alpha_reg = alpha_reg
        self.main_network = main_network
        self.alpha_bias_random_mean = alpha_bias_random_mean
        self.alpha_random_stddev = alpha_random_stddev
        self.beta_random_stddev = beta_random_stddev

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

        # features
        self.x = tf.compat.v1.placeholder(tf.float32, input_shape, name='x')

        # time to event
        self.t = tf.compat.v1.placeholder(tf.float32, [None, 1], name='t')

        # event label
        self.y = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y')

        self.target = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target')
        self.sample_weight = tf.compat.v1.placeholder(tf.float32, [None, 1], name='sample_weight')

        self.o1 = None
        self.set_outputs()
        self.alphas = None
        self.betas = None
        self.main_loss = None
        self.loss = self.calc_loss()

    def set_outputs(self):
        output, output_shape = self.main_network(input_tensor=self.x, seed=self.seed)
        self.o1 = self.layer_weibull_parameters(output, output_shape)

    def layer_weibull_parameters(self, input_t, input_shape):
        # alpha weibull parameter
        alpha_weights = tf.Variable(tf.random.normal(shape=[input_shape[1], 1], seed=self.seed,
                                                     stddev=self.alpha_random_stddev), name='alpha_weight')
        alpha_bias = tf.Variable(tf.random.normal(shape=[1], stddev=self.alpha_random_stddev,
                                                  mean=self.alpha_bias_random_mean, seed=self.seed),
                                 name='alpha_bias')
        alpha = tf.add(tf.matmul(input_t, alpha_weights), alpha_bias, name='alpha_out')
        alpha = tf.clip_by_value(alpha, 0, 12, name='alpha_clipping')
        alpha = tf.exp(alpha, name='alpha_act')
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1), name='alpha_reshaped')
        # beta weibull parameter
        beta_weights = tf.Variable(tf.random.normal(shape=[input_shape[1], 1], stddev=self.beta_random_stddev,
                                                    seed=self.seed), name='beta_weight')
        beta_bias = tf.Variable(tf.random.normal(shape=[1], stddev=self.beta_random_stddev, seed=self.seed),
                                name='beta_bias')
        beta = tf.add(tf.matmul(input_t, beta_weights), beta_bias, name='beta_out')
        beta = tf.clip_by_value(beta, 0, 2, name='beta_clipping')
        beta = tf.nn.softplus(beta, name='beta_act')
        beta = tf.reshape(beta, (tf.shape(beta)[0], 1), name='beta_reshaped')
        # concat weibull parameters
        output = tf.concat((alpha, beta), axis=1, name='wp_concat')
        return output

    def calc_loss(self):
        self.main_loss = self.get_survival_loss()
        return self.main_loss

    def get_survival_loss(self):
        # calculate weibull log likelihood
        sh = tf.shape(self.t)
        self.alphas = tf.reshape(self.o1[:, 0], sh, name='alpha_reshaped_loss')
        self.betas = tf.reshape(self.o1[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for first sample
        mean_lh = losses.weibull_loglikelyhood_loss(self.t, self.y, self.alphas, self.betas)
        # alpha regularizer
        mean_sq_alpha = tf.reduce_mean(self.alphas)
        return mean_lh + self.alpha_reg * mean_sq_alpha


class ContrastiveRankingModel(WeibullModel):

    def __init__(self, input_shape, main_network, seed=7, alpha_reg=1e-3, contrastive_weight=1, margin_weight=1,
                 alpha_bias_random_mean=0.0, alpha_random_stddev=1.0, beta_random_stddev=1.0):
        self.contrastive_weight = contrastive_weight
        self.margin_weight = margin_weight
        self.o1_transformed = None
        self.additional_loss = None
        super().__init__(input_shape, main_network, seed, alpha_reg, alpha_bias_random_mean,
                         alpha_random_stddev, beta_random_stddev)

    def set_outputs(self):
        output, output_shape = self.main_network(input_tensor=self.x, seed=self.seed)
        self.o1 = self.layer_weibull_parameters(output, output_shape)
        # linear transformation
        w_wb = tf.Variable(tf.random_normal(shape=[2, 2], seed=self.seed), name='par_transform')
        self.o1_transformed = tf.matmul(self.o1, w_wb, name='transformed_out')

    def calc_loss(self):
        self.main_loss = self.get_survival_loss()
        hardest_pos_dist, hardest_neg_dist_margin, mean_contr_loss = losses.batch_hard_sampling_contrastive_loss(
            self.o1_transformed, self.target, self.t, self.y, self.margin_weight)
        self.additional_loss = self.contrastive_weight * mean_contr_loss
        return tf.add(self.main_loss, self.additional_loss, name='sum_losses')


class BinaryRankingModel(WeibullModel):

    def __init__(self, input_shape, main_network, seed=7, alpha_reg=1e-3, cross_entropy_weight=1,
                 alpha_bias_random_mean=0.0, alpha_random_stddev=1.0, beta_random_stddev=1.0):
        self.cross_entropy_weight = cross_entropy_weight
        self.additional_loss = None
        super().__init__(input_shape, main_network, seed, alpha_reg, alpha_bias_random_mean,
                         alpha_random_stddev, beta_random_stddev)

    def calc_loss(self):
        self.main_loss = self.get_survival_loss()
        hardest_pos, hardest_neg, mean_ce_loss = losses.batch_hard_sampling_cross_entropy_loss(
            self.o1, self.target, self.t, self.y)
        self.additional_loss = self.cross_entropy_weight * mean_ce_loss
        return tf.add(self.main_loss, self.additional_loss, name='sum_losses')
