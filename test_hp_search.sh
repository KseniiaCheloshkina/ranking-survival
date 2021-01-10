python3.7 hp_search.py \
  --save_prediction True \
  --save_losses True  \
  --save_path 'hp/metabric_base/' \
  --verbose 1 \
  --model_type 'base' \
  --custom_bottom_function_name "metabric_main_network" \
  --config_path "configs/config_metabric_base_hp.json" \
  --train_data_paths "data/input_metabric/metabric_preprocessed_cv_0_train.pkl" "data/input_metabric/metabric_preprocessed_cv_1_train.pkl" "data/input_metabric/metabric_preprocessed_cv_2_train.pkl" "data/input_metabric/metabric_preprocessed_cv_3_train.pkl" "data/input_metabric/metabric_preprocessed_cv_4_train.pkl"\
  --val_data_paths "data/input_metabric/metabric_preprocessed_cv_0_test.pkl" "data/input_metabric/metabric_preprocessed_cv_1_test.pkl" "data/input_metabric/metabric_preprocessed_cv_2_test.pkl" "data/input_metabric/metabric_preprocessed_cv_3_test.pkl" "data/input_metabric/metabric_preprocessed_cv_4_test.pkl"
