import tensorflow as tf


def metabric_main_network(input_tensor, seed):
    n_units = 4
    output_tensor = tf.layers.dense(inputs=input_tensor, units=n_units,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
                                    bias_initializer=tf.keras.initializers.glorot_normal(seed=seed + 1),
                                    name='main_network_dense')
    output_shape = [None, n_units]
    return output_tensor, output_shape


def kkbox_main_network(input_tensor, seed, units_in_layers=[64, 32, 16], dropout=0):

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
