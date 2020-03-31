import keras.backend as K
from keras.models import Model
from keras import layers

def custom_activation(wb_params):
    
    alpha = K.exp(wb_params[:, 0])
    beta = K.softplus(wb_params[:, 1])
    
    beta = K.reshape(beta, (K.shape(beta)[0], 1))
    alpha = K.reshape(alpha, (K.shape(alpha)[0], 1))
    
    return K.concatenate((alpha, beta), axis=1)

def WeibullRankingModel(input_shape, common_nn_input, common_nn_output, loss, optimizer):
    
    # build Siamese network    
    wp = layers.Dense(2, activation=None, name='weibull_params')(common_nn_output)
    wp = layers.Activation(custom_activation)(wp)
    base_network = Model(common_nn_input, wp)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    merged_vector = layers.concatenate([processed_a, processed_b], axis=-1)
    model = Model([input_a, input_b], merged_vector)
    
    # complilation
    if loss == 'binary_ce':
        loss_f = binary_cross_entropy_survival
        batch_generator = binary_cross_entropy_batch_generator
    model.compile(optimizer=optimizer, loss=loss_f)
    
    return model, batch_generator
