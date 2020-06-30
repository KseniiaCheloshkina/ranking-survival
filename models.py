import random

import numpy as np
import tensorflow as tf

import losses


def metabric_main_network(input_tensor, seed):
    n_units = 4
    output_tensor = tf.layers.dense(inputs=input_tensor, units=n_units,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                    bias_initializer=tf.keras.initializers.glorot_normal(seed=seed + 1),
                                    name='main_network_dense')
    output_shape = [None, n_units]
    return output_tensor, output_shape


def kkbox_main_network(input_tensor, units_in_layers, dropout, seed):

    """
    :param input_tensor: Tensor for input to the model
    :param units_in_layers: List(int) - list of number of nodes per layer
    :param dropout: float specifying dropout proba
    :param seed: seed for weights initializations
    """
    # entity embeddings
    gender_matrix = tf.Variable(tf.random.normal(shape=[2, 1], seed=seed), name='gender_matrix')
    gender_na_bias = tf.Variable(tf.random.normal(shape=[1], seed=seed), name='gender_na_bias')
    gender_embed = tf.add(tf.matmul(input_tensor[:, 0:2], gender_matrix, name='gender_embed'),
                          gender_na_bias, name='gender_out')

    city_matrix = tf.Variable(tf.random.normal(shape=[21, 4], seed=seed), name='city_matrix')
    city_na_bias = tf.Variable(tf.random.normal(shape=[1], seed=seed), name='city_na_bias')
    city_embed = tf.add(tf.matmul(input_tensor[:, 2:23], city_matrix, name='city_embed'),
                        city_na_bias, name='city_out')

    reg_matrix = tf.Variable(tf.random.normal(shape=[5, 2], seed=seed), name='reg_matrix')
    reg_na_bias = tf.Variable(tf.random.normal(shape=[1], seed=seed), name='reg_na_bias')
    reg_embed = tf.add(tf.matmul(input_tensor[:, 23:28], reg_matrix, name='reg_embed'),
                       reg_na_bias, name='reg_out')

    entity_embed = tf.concat((gender_embed, city_embed, reg_embed), axis=1, name='entity_embed_concat')
    # final input
    data = tf.concat((entity_embed, input_tensor[:, 28:]), axis=1, name='preproc_input')

    for layer_ind, layer_units in enumerate(units_in_layers):
        dense = tf.keras.layers.Dense(units=layer_units, activation='relu',
                                      kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                      bias_initializer=tf.keras.initializers.glorot_normal(seed=seed + 1),
                                      name='dense_layer_' + str(layer_ind))(inputs=data)
        data = tf.nn.dropout(dense, rate=dropout, seed=seed, name='dropout_' + str(layer_ind))

    output_shape = [None, layer_units]
    return data, output_shape


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
        self.x_a = tf.compat.v1.placeholder(tf.float32, input_shape, name='x_a')
        self.x_b = tf.compat.v1.placeholder(tf.float32, input_shape, name='x_b')

        # time to event
        self.t_a = tf.compat.v1.placeholder(tf.float32, [None, 1], name='t_a')
        self.t_b = tf.compat.v1.placeholder(tf.float32, [None, 1], name='t_b')

        # event label
        self.y_a = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_a')
        self.y_b = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_b')

        self.target = tf.compat.v1.placeholder(tf.float32, [None, 1], name='target')
        self.sample_weight = tf.compat.v1.placeholder(tf.float32, [None, 1], name='sample_weight')

        self.o1 = None
        self.o2 = None
        self.set_outputs()
        self.alphas_a = None
        self.betas_a = None
        self.alphas_b = None
        self.betas_b = None
        self.loss = self.calc_loss()

    def set_outputs(self):
        with tf.compat.v1.variable_scope("siamese", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            self.o1 = self.siamese_net(self.x_a)
            scope.reuse_variables()
            self.o2 = self.siamese_net(self.x_b)

    def layer_weibull_parameters(self, input_t, input_shape):
        # alpha weibull parameter
        alpha_weights = tf.Variable(tf.random.normal(shape=[input_shape[1], 1], seed=self.seed,
                                                     stddev=self.alpha_random_stddev), name='alpha_weight')
        alpha_bias = tf.Variable(tf.random.normal(shape=[1], seed=self.seed, stddev=self.alpha_random_stddev,
                                                  mean=self.alpha_bias_random_mean), name='alpha_bias')
        alpha = tf.add(tf.matmul(input_t, alpha_weights), alpha_bias, name='alpha_out')
        alpha = tf.clip_by_value(alpha, 0, 12, name='alpha_clipping')
        alpha = tf.exp(alpha, name='alpha_act')
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1), name='alpha_reshaped')
        # beta weibull parameter
        beta_weights = tf.Variable(tf.random.normal(shape=[input_shape[1], 1], seed=self.seed,
                                                    stddev=self.beta_random_stddev), name='beta_weight')
        beta_bias = tf.Variable(tf.random.normal(shape=[1], seed=self.seed, stddev=self.beta_random_stddev),
                                name='beta_bias')
        beta = tf.add(tf.matmul(input_t, beta_weights), beta_bias, name='beta_out')
        beta = tf.clip_by_value(beta, 0, 2, name='beta_clipping')
        beta = tf.nn.softplus(beta, name='beta_act')
        beta = tf.reshape(beta, (tf.shape(beta)[0], 1), name='beta_reshaped')
        # concat weibull parameters
        output = tf.concat((alpha, beta), axis=1, name='wp_concat')
        return output

    def siamese_net(self, x):
        output, output_shape = self.main_network(input_tensor=x, seed=self.seed)
        output = self.layer_weibull_parameters(output, output_shape)
        return output
    
    def calc_loss(self):
        return self.get_survival_loss()

    def get_survival_loss(self):
        sh = tf.shape(self.t_a)
        self.alphas_a = tf.reshape(self.o1[:, 0], sh, name='alpha_reshaped_loss')
        self.betas_a = tf.reshape(self.o1[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for first sample
        mean_lh_a = losses.weibull_loglikelyhood_loss(self.t_a, self.y_a, self.alphas_a, self.betas_a)

        self.alphas_b = tf.reshape(self.o2[:, 0], sh, name='alpha_reshaped_loss')
        self.betas_b = tf.reshape(self.o2[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for second sample
        mean_lh_b = losses.weibull_loglikelyhood_loss(self.t_b, self.y_b, self.alphas_b, self.betas_b)

        # alpha regularizer
        all_alphas = tf.add(self.alphas_a, self.alphas_b, name='survival_loss_alpha_beta_sum')
        mean_sq_alpha = tf.reduce_mean(all_alphas)
        self.mean_sq_alpha = mean_sq_alpha
        self.mean_lh_a = mean_lh_a
        self.mean_lh_b = mean_lh_b
        return mean_lh_b + mean_lh_a + self.alpha_reg * mean_sq_alpha


class BinaryRankingModel(WeibullModel):

    def __init__(self, input_shape, main_network, seed=7, alpha_reg=1e-3, cross_entropy_weight=1,
                 alpha_bias_random_mean=0.0, alpha_random_stddev=1.0, beta_random_stddev=1.0):
        self.cross_entropy_weight = cross_entropy_weight
        super().__init__(input_shape, main_network, seed, alpha_reg, alpha_bias_random_mean,
                         alpha_random_stddev, beta_random_stddev)

    def calc_loss(self):
        main_loss = self.get_survival_loss()
        # binary cross-entropy
        mean_ll = losses.binary_cross_entropy_loss(self.t_a, self.t_b, self.alphas_a, self.betas_a,
                                                   self.alphas_b, self.betas_b, self.target, self.sample_weight)
        return main_loss + self.cross_entropy_weight * mean_ll


class ContrastiveRankingModel(WeibullModel):

    def __init__(self, input_shape, main_network, seed=7, alpha_reg=1e-3, contrastive_weight=1, margin_weight=1,
                 alpha_bias_random_mean=0.0, alpha_random_stddev=1.0, beta_random_stddev=1.0):
        self.contrastive_weight = contrastive_weight
        self.margin_weight = margin_weight
        self.o1_transformed = None
        self.o2_transformed = None
        super().__init__(input_shape, main_network, seed, alpha_reg, alpha_bias_random_mean,
                         alpha_random_stddev, beta_random_stddev)

    def siamese_net(self, x):
        output, output_shape = self.main_network(input_tensor=x, seed=self.seed)
        output = self.layer_weibull_parameters(output, output_shape)
        # linear transformation
        w_wb = tf.Variable(tf.random_normal(shape=[2, 2], seed=self.seed), name='par_transform')
        output_transform = tf.matmul(output, w_wb, name='transformed_out')
        return output, output_transform

    def set_outputs(self):
        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            self.o1, self.o1_transformed = self.siamese_net(self.x_a)
            scope.reuse_variables()
            self.o2, self.o2_transformed = self.siamese_net(self.x_b)

    def calc_loss(self):
        self.main_loss = self.get_survival_loss()
        # contrastive (margin) loss
        mean_contr_loss = losses.contrastive_margin_loss(self.o1_transformed, self.o2_transformed, self.target,
                                                         self.margin_weight * self.sample_weight)
        return tf.add(self.main_loss, self.contrastive_weight * mean_contr_loss, name='sum_losses')
