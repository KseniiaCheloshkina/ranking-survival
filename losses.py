import keras.backend as K

def binary_cross_entropy_survival(y_true, y_pred):
    
    """
    y_pred (, 4) - output of network
        y_pred[:, 0:2] - weibull parameters for first input
        y_pred[:, 2:4] - weibull parameters for second input
    y_true (, 4) - target
        y_true[:, 0] - t for first input
        y_true[:, 1] - t for second input  
        y_true[:, 2] - target (1 if q1 == q2 and 0 otherwise)
        y_true[:, 3] - sample weight 
    """
    
    # get a probaility    
    t_a = y_true[:, 0]
    t_b = y_true[:, 1]
    o = y_true[:, 2]
    q = y_true[:, 3]
    
    y_pred_a = y_pred[:, 0:2]
    y_pred_b = y_pred[:, 2:4]

    s_a = calc_survival_value(t_a, y_pred_a)
    s_b = calc_survival_value(t_b, y_pred_b)
    sigm = K.sigmoid(s_a - s_b)
    sigm = K.clip(sigm, K.epsilon(), 1 - K.epsilon())
    
    # weighted binary cross entropy
    label_pos = o * K.log(sigm + K.epsilon())
    label_neg = (1 - o) * K.log(1 + K.epsilon() - sigm) * (q + 1)  
    print(s_a)
    print(s_b)
    return -1 * K.mean(label_pos + label_neg)

   
def calc_survival_value(y_true, y_pred):
    
    alphas = y_pred[:, 0]
    betas = y_pred[:, 1]
    # TODO: clipping     sigm = K.clip(sigm, K.epsilon(), 1 - K.epsilon())
    s = K.exp(-1 * K.pow(y_true / (alphas + 1e-6), betas))
              
    return s                                              
