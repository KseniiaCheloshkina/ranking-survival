import argparse
import tensorflow as tf
import copy
import tqdm

import train


def hp_grid():
    # for base model
    alpha_reg_values = [1e-6, 1e-3, 1e-1]
    # for cross-entropy model
    cross_entropy_weight_values = [0.1, 1, 2, 5]
    # for contrastive model
    contrastive_weight_values = [0.1, 0.5, 1, 3, 5]
    return alpha_reg_values, cross_entropy_weight_values, contrastive_weight_values


def main(args):
    cur_args = copy.deepcopy(args)
    del cur_args['config_path_base']
    del cur_args['config_path_contrastive']
    del cur_args['config_path_binary']
    alpha_reg_values, cross_entropy_weight_values, contrastive_weight_values = hp_grid()

    if args['config_path_base'] is not None:
        print("Fitting base model...")
        cur_args.update(
            {
                'model_type': 'base',
                'config_path': args['config_path_base'],
                'save_path': args['save_path'] + 'base/',

                'save_model': True,
                'save_prediction': True,
                'save_losses': True
            }
        )
        train_data, val_data, config, seed = train.load_and_check(cur_args)
        for par in tqdm.tqdm(alpha_reg_values):
            config.update({'alpha_reg': par})
            tf.reset_default_graph()
            cur_args.update({
                'save_path': '{}_alpha_reg_{}_'.format(cur_args['save_path'], str(par))
            })
            train.train_save(cur_args, train_data, val_data, config, seed)

    if args['config_path_binary'] is not None:
        print("Fitting binary model...")
        cur_args.update(
            {
                'model_type': 'binary',
                'config_path': args['config_path_binary'],
                'save_path': args['save_path'] + 'binary/',

                'save_model': True,
                'save_prediction': True,
                'save_losses': True
            }
        )
        train_data, val_data, config, seed = train.load_and_check(cur_args)
        for par in tqdm.tqdm(cross_entropy_weight_values):
            config.update({'cross_entropy_weight': par})
            tf.reset_default_graph()
            cur_args.update({
                'save_path': '{}_ce_weight_{}_'.format(cur_args['save_path'], str(par))
            })
            train.train_save(cur_args, train_data, val_data, config, seed)

    if args['config_path_contrastive'] is not None:
        print("Fitting contrastive model...")
        cur_args.update(
            {
                'model_type': 'contrastive',
                'config_path': args['config_path_contrastive'],
                'save_path': args['save_path'] + 'contrastive/',

                'save_model': True,
                'save_prediction': True,
                'save_losses': True
            }
        )
        train_data, val_data, config, seed = train.load_and_check(cur_args)
        for par in tqdm.tqdm(contrastive_weight_values):
            config.update({'contrastive_weight': par})
            tf.reset_default_graph()
            cur_args.update({
                'save_path': '{}_contr_weight_{}_'.format(cur_args['save_path'], str(par))
            })
            train.train_save(cur_args, train_data, val_data, config, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model on given dataset')
    # running options
    # TODO: type of y
    parser.add_argument('--train_data_path', required=True, type=str,
                        help='''
                        Path to pickle file to train model on. 
                        Format: pickle file contains a dictionary with 3 keys:
                        x - features numpy array (n_samples, n_features)
                        y - event label numpy array (n_samples, )
                        t - duration label numpy array of type float (n_samples, )
                        ''')
    parser.add_argument('--val_data_path', required=False, type=str,
                        help='''
                        Path to pickle file to validate model on
                        Format: pickle file contains a dictionary with 3 keys:
                        x - features numpy array (n_samples, n_features)
                        y - event label numpy array (n_samples, )
                        t - duration label numpy array of type float (n_samples, )
                        ''')
    parser.add_argument('--config_path_base', required=True, type=str, default=None,
                        help='Path to json file with all the parameters values for BASE model')
    parser.add_argument('--config_path_contrastive', required=True, type=str, default=None,
                        help='Path to json file with all the parameters values for CONTRASTIVE model')
    parser.add_argument('--config_path_binary', required=True, type=str, default=None,
                        help='Path to json file with all the parameters values for CROSS_ENTROPY model')
    parser.add_argument('--custom_bottom_function_name', required=True, type=str,
                        help='Name of function from custom_models.py to use as bottom for model. '
                             'To this network ""survival" head will be appended')
    parser.add_argument('--verbose', required=False, type=int, choices=[0, 1], default=0,
                        help='Whether to print detailed stats')
    # saving options
    parser.add_argument('--save_path', required=False, type=str, default='tmp/',
                        help='Path to store data in case of save_model or save_prediction options is on')
    arguments = vars(parser.parse_args())
    print("Arguments: ")
    print(arguments)
    main(arguments)
