import os
import sys
import pickle
from tqdm.notebook import tqdm as tqdm
import random

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

sys.path.append("../")

from batch_generators import ContrastiveDataGenerator, BinaryDataGenerator
from models import ContrastiveRankingModel, metabric_main_network
from tools import test_quality
from clr import cyclic_learning_rate

s = 2

random.seed(s)
np.random.seed(s)
tf.set_random_seed(s)
os.environ['PYTHONHASHSEED'] = str(s)
os.environ['TF_CUDNN_DETERMINISTIC'] = str(s)


best_metabric_params = {
    'n_time_bins': 10,
    'inp_shape': (None, 9),
    'n_epochs_binary': 30,
    'n_epochs_contrastive_freezed': 60,
    'n_epochs_contrastive': 120,
    'seed': s,
    'max_lr': 0.002,
    'step_size': 13,
    'time_grid': np.linspace(0, 300, 30, dtype=np.int),
    'learning_rate_contr_freezed': 0.0001,
    'learning_rate_contr': 0.00005,
    'momentum': 0.7,
    'batch_size': 1024,
    'batch_size_contr': 1024,
    'alpha_reg': 1e-6,
    'margin_weight': 1e-1,
    'contrastive_weight': 0.5
}


# concordance at - one of points in time_grid
def train_model(train_data, test_data, model_params):
    tf.reset_default_graph()
    model = ContrastiveRankingModel(input_shape=model_params['inp_shape'], seed=model_params['seed'], alpha_reg=model_params['alpha_reg'], 
                                    main_network=metabric_main_network, 
                                    contrastive_weight=model_params['contrastive_weight'], margin_weight=model_params['margin_weight'])
    trainable_binary = [var for var in tf.global_variables() if 'transform' not in var.name]
    trainable_contr = [var for var in tf.global_variables() if 'transform' in var.name]
    trainable_contr_binary = tf.global_variables()
    ## contrastive data generator
    # validation data
    dg = ContrastiveDataGenerator(x=test_data['x'], y=test_data['y'], t=test_data['t'], batch_size=model_params['batch_size_contr'], 
                                  n_time_bins=model_params['n_time_bins'])
    val_size = min(dg.ij_pos_sorted.shape[1], dg.ij_neg_sorted.shape[1])
    dg = ContrastiveDataGenerator(x=test_data['x'], y=test_data['y'], t=test_data['t'], batch_size=val_size, n_time_bins=model_params['n_time_bins'])
    [x_batch_left_val_contrastive, x_batch_right_val_contrastive], y_batch_val_contrastive, sample_weight_val_contrastive, target_val_contrastive = next(
        dg.get_batch())
    val_size_contrastive = x_batch_left_val_contrastive.shape[0]
    # training data
    dg_contrastive = ContrastiveDataGenerator(x=train_data['x'], y=train_data['y'], t=train_data['t'], batch_size=model_params['batch_size_contr'],
                             n_time_bins=model_params['n_time_bins'])    
    n_batches_contrastive = min(dg_contrastive.ij_pos_sorted.shape[1], dg_contrastive.ij_neg_sorted.shape[1]) // model_params['batch_size_contr']
    
    
    n_epochs_contrastive = model_params['n_epochs_contrastive']
    n_epochs_contrastive_freezed = model_params['n_epochs_contrastive_freezed']
    
    ## binary data generator    
    # validation data
    dg = BinaryDataGenerator(x=test_data['x'], y=test_data['y'], t=test_data['t'], batch_size=model_params['batch_size'], n_time_bins=model_params['n_time_bins'])
    val_size = min(dg.ij_pos_sorted.shape[1], dg.ij_neg_sorted.shape[1])
    dg_binary = BinaryDataGenerator(x=test_data['x'], y=test_data['y'], t=test_data['t'], batch_size=val_size, n_time_bins=model_params['n_time_bins'])
    [x_batch_left_val_binary, x_batch_right_val_binary], y_batch_val_binary, sample_weight_val_binary, target_val_binary = next(dg.get_batch())
    val_size_binary = x_batch_left_val_binary.shape[0]
    # training data
    dg_binary = BinaryDataGenerator(x=train_data['x'], y=train_data['y'], t=train_data['t'], batch_size=model_params['batch_size'],
                             n_time_bins=model_params['n_time_bins'])
    n_batches_binary = min(dg_binary.ij_pos_sorted.shape[1], dg_binary.ij_neg_sorted.shape[1]) // model_params['batch_size']
    n_epochs_binary = model_params['n_epochs_binary']
    
    # optimizators
    # cycling learning rate
    all_lr = []
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = cyclic_learning_rate(global_step=global_step, learning_rate=model_params['max_lr'] / 5, max_lr=model_params['max_lr'], 
                                         step_size=model_params['step_size'], mode='triangular2')

    
    all_pred = []
    train_loss = []
    val_loss = []
    train_contr_loss = []
    val_contr_loss = []
    # Launch the graph
    with tf.device('/GPU:0'):
        with tf.Session() as sess:
            tf.set_random_seed(model_params['seed'])
            
            optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op_weibull = optimizer1.minimize(model.main_loss, global_step=global_step, var_list=trainable_binary)
            optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=model_params['learning_rate_contr_freezed'])
            train_op_contrastive_freezed = optimizer3.minimize(model.loss, var_list=trainable_contr)  
            optimizer2 = tf.train.MomentumOptimizer(learning_rate=model_params['learning_rate_contr'], momentum=model_params['momentum'])
            train_op_contrastive = optimizer2.minimize(model.loss, var_list=trainable_contr_binary)
            
            init = tf.initialize_all_variables()
            sess.run(init)
            
            ## train standard weibull loss on binary batch generator
            for i in tqdm(range(n_epochs_binary)):
                # initialize generator
                gen = dg_binary.get_batch()
                # for each batch
                for j in range(n_batches_binary):
                    # get batch data
                    [x_batch_left, x_batch_right], y_batch, sample_weight, target = next(gen)
                    feed_dict = {
                        model.x_a: x_batch_left, 
                        model.x_b: x_batch_right, 
                        model.t_a: y_batch[:, 0].reshape((model_params['batch_size'], 1)),
                        model.t_b: y_batch[:, 1].reshape((model_params['batch_size'], 1)),
                        model.y_a: y_batch[:, 2].reshape((model_params['batch_size'], 1)),
                        model.y_b: y_batch[:, 3].reshape((model_params['batch_size'], 1)),
                        model.target: target.reshape((model_params['batch_size'], 1)),
                        model.sample_weight: sample_weight
                    }
                    
                    # train model
                    _, l = sess.run([train_op_weibull, model.main_loss], feed_dict=feed_dict)
                    
                 # change learning rate
                assign_op = global_step.assign(i)
                sess.run(assign_op)               
                new_lr = sess.run(optimizer1._learning_rate_tensor)
                all_lr.append(new_lr)
                
                # get predictions for validation data
                pred = sess.run(model.o1, feed_dict={model.x_a: test_data['x']})
                all_pred.append(pred)
                # save train loss
                train_loss.append(l)
                # save test loss
                l = sess.run([model.main_loss], feed_dict={
                    model.x_a: x_batch_left_val_binary, 
                    model.x_b: x_batch_right_val_binary, 
                    model.t_a: y_batch_val_binary[:, 0].reshape((val_size_binary, 1)),
                    model.t_b: y_batch_val_binary[:, 1].reshape((val_size_binary, 1)),
                    model.y_a: y_batch_val_binary[:, 2].reshape((val_size_binary, 1)),
                    model.y_b: y_batch_val_binary[:, 3].reshape((val_size_binary, 1)),
                    model.target: target_val_binary.reshape((val_size_binary, 1)),
                    model.sample_weight: sample_weight_val_binary
                }) 
                val_loss.append(l)
                

            ## train contrastive loss on contrastive batch generator FREEZED
            for i in tqdm(range(n_epochs_contrastive_freezed)):
                # initialize generator
                gen = dg_contrastive.get_batch()
                # for each batch
                for j in range(n_batches_contrastive):
                    # get batch data
                    [x_batch_left, x_batch_right], y_batch, sample_weight, target = next(gen)
                    feed_dict = {
                        model.x_a: x_batch_left, 
                        model.x_b: x_batch_right, 
                        model.t_a: y_batch[:, 0].reshape((model_params['batch_size_contr'], 1)),
                        model.t_b: y_batch[:, 1].reshape((model_params['batch_size_contr'], 1)),
                        model.y_a: y_batch[:, 2].reshape((model_params['batch_size_contr'], 1)),
                        model.y_b: y_batch[:, 3].reshape((model_params['batch_size_contr'], 1)),
                        model.target: target.reshape((model_params['batch_size_contr'], 1)),
                        model.sample_weight: sample_weight
                    }
                    
                    # train model
                    _, l, l_c = sess.run([train_op_contrastive_freezed, model.main_loss, model.loss], feed_dict=feed_dict)
                
                # get predictions for validation data
                pred = sess.run(model.o1, feed_dict={model.x_a: test_data['x']})
                all_pred.append(pred)
                # save train loss
                train_loss.append(l)
                train_contr_loss.append(l_c)
                # save test loss
                l, l_c = sess.run([model.main_loss, model.loss], feed_dict={
                    model.x_a: x_batch_left_val_contrastive, 
                    model.x_b: x_batch_right_val_contrastive, 
                    model.t_a: y_batch_val_contrastive[:, 0].reshape((val_size_contrastive, 1)),
                    model.t_b: y_batch_val_contrastive[:, 1].reshape((val_size_contrastive, 1)),
                    model.y_a: y_batch_val_contrastive[:, 2].reshape((val_size_contrastive, 1)),
                    model.y_b: y_batch_val_contrastive[:, 3].reshape((val_size_contrastive, 1)),
                    model.target: target_val_contrastive.reshape((val_size_contrastive, 1)),
                    model.sample_weight: sample_weight_val_contrastive
                }) 
                val_loss.append(l)     
                val_contr_loss.append(l_c)
                
                
                
                
            ## train contrastive loss on contrastive batch generator
            for i in tqdm(range(n_epochs_contrastive)):
                # initialize generator
                gen = dg_contrastive.get_batch()
                # for each batch
                for j in range(n_batches_contrastive):
                    # get batch data
                    [x_batch_left, x_batch_right], y_batch, sample_weight, target = next(gen)
                    feed_dict = {
                        model.x_a: x_batch_left, 
                        model.x_b: x_batch_right, 
                        model.t_a: y_batch[:, 0].reshape((model_params['batch_size_contr'], 1)),
                        model.t_b: y_batch[:, 1].reshape((model_params['batch_size_contr'], 1)),
                        model.y_a: y_batch[:, 2].reshape((model_params['batch_size_contr'], 1)),
                        model.y_b: y_batch[:, 3].reshape((model_params['batch_size_contr'], 1)),
                        model.target: target.reshape((model_params['batch_size_contr'], 1)),
                        model.sample_weight: sample_weight
                    }
                    
                    # train model
                    _, l, l_c = sess.run([train_op_contrastive, model.main_loss, model.loss], feed_dict=feed_dict)
                
                # get predictions for validation data
                pred = sess.run(model.o1, feed_dict={model.x_a: test_data['x']})
                all_pred.append(pred)
                # save train loss
                train_loss.append(l)
                train_contr_loss.append(l_c)
                # save test loss
                l, l_c = sess.run([model.main_loss, model.loss], feed_dict={
                    model.x_a: x_batch_left_val_contrastive, 
                    model.x_b: x_batch_right_val_contrastive, 
                    model.t_a: y_batch_val_contrastive[:, 0].reshape((val_size_contrastive, 1)),
                    model.t_b: y_batch_val_contrastive[:, 1].reshape((val_size_contrastive, 1)),
                    model.y_a: y_batch_val_contrastive[:, 2].reshape((val_size_contrastive, 1)),
                    model.y_b: y_batch_val_contrastive[:, 3].reshape((val_size_contrastive, 1)),
                    model.target: target_val_contrastive.reshape((val_size_contrastive, 1)),
                    model.sample_weight: sample_weight_val_contrastive
                }) 
                val_loss.append(l)     
                val_contr_loss.append(l_c)

    return all_pred, train_loss, val_loss, train_contr_loss, val_contr_loss, all_lr
