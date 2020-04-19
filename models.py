import tensorflow as tf
import losses


class Model(object):

    def __init__(self, input_shape=(None, 9)):
        # TODO: add input_shape for features to init arguments
        # TODO: add main network to init arguments
        # features
        self.x_a = tf.placeholder(tf.float32, input_shape)
        self.x_b = tf.placeholder(tf.float32, input_shape)

        # time to event
        self.t_a = tf.placeholder(tf.float32, [None, 1], name='t_a')
        self.t_b = tf.placeholder(tf.float32, [None, 1], name='t_b')

        # event label
        self.y_a = tf.placeholder(tf.float32, [None, 1], name='y_a')
        self.y_b = tf.placeholder(tf.float32, [None, 1], name='y_b')

        self.target = tf.placeholder(tf.float32, [None, 1], name='target')
        self.sample_weight = tf.placeholder(tf.float32, [None, 1], name='sample_weight')

        with tf.variable_scope("siamese", reuse=tf.AUTO_REUSE) as scope:
            self.o1 = self.siamese_net(self.x_a)
            scope.reuse_variables()
            self.o2 = self.siamese_net(self.x_b)

        self.loss = self.calc_loss()

    def siamese_net(self, x):
        # main network
        output = tf.layers.dense(inputs=x, units=4, name='dense')

        # weibull parameters layer
        init_kernel = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        y_pred = tf.layers.dense(inputs=output, units=2, bias_initializer=init_kernel, kernel_initializer=init_kernel,
                                 name='weibull_parameters')
        alpha = y_pred[:, 0]
        beta = y_pred[:, 1]
        alpha = tf.clip_by_value(alpha, 0, 12, name='alpha_clipping')
        alpha = tf.exp(alpha, name='alpha_act')
        beta = tf.clip_by_value(beta, 0, 2, name='beta_clipping')
        beta = tf.nn.softplus(beta, name='beta_act')
        beta = tf.reshape(beta, (tf.shape(beta)[0], 1), name='beta_reshaped')
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1), name='alpha_reshaped')
        output = tf.concat((alpha, beta), axis=1, name='wp_concat')
        return output

    def calc_loss(self):
        pass


class BinaryRankingModel(Model):

    def calc_loss(self):

        sh = tf.shape(self.t_a)
        alphas_a = tf.reshape(self.o1[:, 0], sh, name='alpha_reshaped_loss')
        betas_a = tf.reshape(self.o1[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for first sample
        mean_lh_a = losses.weibull_loglikelyhood_loss(self.t_a, self.y_a, alphas_a, betas_a)

        alphas_b = tf.reshape(self.o2[:, 0], sh, name='alpha_reshaped_loss')
        betas_b = tf.reshape(self.o2[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for second sample
        mean_lh_b = losses.weibull_loglikelyhood_loss(self.t_b, self.y_b, alphas_b, betas_b)

        # alpha regularizer
        all_alphas = tf.add(alphas_a, alphas_b)
        mean_sq_alpha = tf.reduce_mean(all_alphas)

        # binary cross-entropy
        mean_ll = losses.binary_cross_entropy_loss(self.t_a, self.t_b, alphas_a, betas_a, alphas_b, betas_b,
                                                   self.target, self.sample_weight)

        return mean_lh_b + mean_lh_a + 1e-3 * mean_sq_alpha + mean_ll


class WeibullModel(Model):

    def calc_loss(self):

        sh = tf.shape(self.t_a)
        alphas_a = tf.reshape(self.o1[:, 0], sh, name='alpha_reshaped_loss')
        betas_a = tf.reshape(self.o1[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for first sample
        mean_lh_a = losses.weibull_loglikelyhood_loss(self.t_a, self.y_a, alphas_a, betas_a)

        alphas_b = tf.reshape(self.o2[:, 0], sh, name='alpha_reshaped_loss')
        betas_b = tf.reshape(self.o2[:, 1], sh, name='beta_reshaped_loss')
        # weibull log likelihood for second sample
        mean_lh_b = losses.weibull_loglikelyhood_loss(self.t_b, self.y_b, alphas_b, betas_b)

        # alpha regularizer
        all_alphas = tf.add(alphas_a, alphas_b)
        mean_sq_alpha = tf.reduce_mean(all_alphas)

        return mean_lh_b + mean_lh_a + 1e-3 * mean_sq_alpha
