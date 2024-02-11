import os
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import sys
import xgboost as xg

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the lib folder to the path

from kfold import Kfold
from lib.augmentation_methods import AugmentationMethods
from lib.matlab_processor import MatlabProcessor
from lib.data.improve import ImproveData
from lib.data.preprocessor import DataPreprocessor
from lib.graphics import Graphics


#  Process the arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Process some arguments.")

parser.add_argument('--apply-data-augmentation', action='store_true', help='Apply data augmentation to the dataset')
parser.add_argument('--apply-min-max-normalization', action='store_true', help='Apply Min-Max normalization to the dataset')
parser.add_argument('--apply-kalman-filter', action='store_true', help='Apply Kalman filter to the dataset')
parser.add_argument('--include-shank-angles', action='store_true', help='Include ShankAngles from the dataset')
parser.add_argument('--include-thigh-angles', action='store_true', help='Include ThighAngles from the dataset')
parser.add_argument('--include-non-linear-data', action='store_true', help='Include non-linear data in the dataset')

args = parser.parse_args()

apply_data_augmentation = args.apply_data_augmentation
apply_min_max_normalization = args.apply_min_max_normalization
apply_kalman_filter = args.apply_kalman_filter
include_shank_angles = args.include_shank_angles
include_thigh_angles = args.include_thigh_angles
include_non_linear_data = args.include_non_linear_data

#  Quick check
if not include_shank_angles and not include_thigh_angles:
    print("!! WARNING !!")
    raise ValueError("At least one of the following flags must be True: include_shank_angles, include_thigh_angles")

#  Print the flags that were passed
print("apply_data_augmentation: ", apply_data_augmentation)
print("apply_min_max_normalization: ", apply_min_max_normalization)
print("apply_kalman_filter: ", apply_kalman_filter)
print("include_shank_angles: ", include_shank_angles)
print("include_thigh_angles: ", include_thigh_angles)
print("include_non_linear_data: ", include_non_linear_data)

print("")

#  This is just to generate some sort of CSV
# -----------------------------------------------------------------------------
print("Person,Train RMSE,Test RMSE,Train R2,Test R2")


#  Process all people
# -----------------------------------------------------------------------------
time_start = time.time()
rmse_test = []

for person_number in range(1, 22):
    # We have 6 datasets, 3 for the shank and 3 for the thigh (05ms, 10ms, 15ms)
    file_names = ['ShkAngW_05ms.mat', 'ShkAngW_10ms.mat', 'ShkAngW_15ms.mat', 'ThiAngW_05ms.mat', 'ThiAngW_10ms.mat', 'ThiAngW_15ms.mat']
    datasets = [MatlabProcessor.to_df(file_name) for file_name in file_names]


    #  DataFrame concatenation depending on the flags
    # -------------------------------------------------------------------------
    independent_variable_columns = []
    if include_shank_angles and include_thigh_angles:
        independent_variable_columns = ['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']
        shankDF = pd.concat(datasets[:3], axis=0)
        thighDF = pd.concat(datasets[3:], axis=0).drop('gait_percentage', axis=1)
    elif include_shank_angles:
        independent_variable_columns = ['ShankAngles', 'ShankAngularVelocity']
        shankDF = pd.concat(datasets[:3], axis=0)
        thighDF = pd.DataFrame()
    elif include_thigh_angles:
        independent_variable_columns = ['ThighAngles', 'ThighAngularVelocity']
        shankDF = pd.DataFrame()
        thighDF = pd.concat(datasets[3:], axis=0)

    k_fold_df = pd.concat([shankDF, thighDF], axis=1)


    #  Add preprocessing methods to the dataset
    # -------------------------------------------------------------------------
    if apply_data_augmentation:
        k_fold_df = AugmentationMethods().augment_dataset(k_fold_df, include_shank=include_shank_angles, include_thigh=include_thigh_angles)

    if apply_kalman_filter:
        k_fold_df = DataPreprocessor().kalman_filter(k_fold_df, independent_variable_columns)

    if apply_min_max_normalization:
        k_fold_df = DataPreprocessor().min_max_normalization(k_fold_df, independent_variable_columns)

    k_fold_df, k_fold_person_df  = Kfold().remove_k_persons(k_fold_df, person_number)
    k_fold_df = ImproveData().add_non_linear_data(k_fold_df)       # Suggested by Mahdy
    k_fold_person_df = ImproveData().add_non_linear_data(k_fold_person_df)

    if(include_non_linear_data):
        independent_variable_columns += ['non_linear_1', 'non_linear_2']

    train_X = k_fold_df[independent_variable_columns]
    test_X = k_fold_person_df[independent_variable_columns]
    train_y = k_fold_df['gait_percentage']
    test_y = k_fold_person_df['gait_percentage']

    # ---------------------------------------------------------------------------------------------
    # reg:squarederror =>  The graphics looks good, but at the beginning or the end of the gait cycle,
    #                      prediction looks wobbly. The figures below are not that bad.
    #
    # Time taken: 62.141404151916504 seconds
    # Average RMSE:  9.478249227216725
    # CV RMSE:  0.319410408736641
    # Standard Deviation RMSE:  3.027451459773046
    # Variance RMSE:  9.165462341281946
    # Max RMSE:  16.35615299769192
    # Min RMSE:  4.785775098485626
    # Median RMSE:  9.87174405117115
    #
    # best_xgb_r = xg.XGBRegressor(objective='reg:squarederror', seed=123, max_leaves=30, learning_rate=0.11, min_child_weight=0.9)

    # ---------------------------------------------------------------------------------------------
    # reg:logistic => not working

    # ---------------------------------------------------------------------------------------------
    # reg:pseudohubererror =>  [NOT EVEN CLOSE] Figures and graphics look awful.
    #
    # Time taken: 60.5547411441803 seconds
    # Average RMSE:  146.39980066737317
    # CV RMSE:  1.9413762382764025e-16
    # Standard Deviation RMSE:  2.842170943040401e-14
    # Variance RMSE:  8.077935669463161e-28
    # Max RMSE:  146.39980066737314
    # Min RMSE:  146.39980066737314
    # Median RMSE:  146.39980066737314
    # best_xgb_r = xg.XGBRegressor(objective='reg:pseudohubererror', seed=123)

    # ---------------------------------------------------------------------------------------------
    # reg:squaredlogerror =>  This doesn't work very well. The figures below are not that bad, but
    #                         prediction looks not like a straight line compared to the identity function.
    #
    # Time taken: 63.10734701156616 seconds
    # Average RMSE:  12.027978908386391
    # CV RMSE:  0.22560693916502111
    # Standard Deviation RMSE:  2.7135955058624854
    # Variance RMSE:  7.363600569437078
    # Max RMSE:  18.040232650908678
    # Min RMSE:  6.108040527942163
    # Median RMSE:  11.747120421442903
    #
    # best_xgb_r = xg.XGBRegressor(objective='reg:squaredlogerror', seed=123, max_leaves=30, learning_rate=0.11, min_child_weight=0.9)

    # ---------------------------------------------------------------------------------------------
    # reg:gamma =>  This is the best one so far. Figures look not tha bad and graphics looks good
    #
    # Time taken: 64.37723588943481 seconds
    # Average RMSE:  9.944619942074864
    # CV RMSE:  0.2897436354161814
    # Standard Deviation RMSE:  2.8813903348490264
    # Variance RMSE:  8.302410261761384
    # Max RMSE:  18.079533899678026
    # Min RMSE:  4.461843626539606
    # Median RMSE:  9.51982554347156
    #
    # best_xgb_r = xg.XGBRegressor(objective='reg:gamma', seed=123, max_leaves=30, learning_rate=0.11, min_child_weight=0.9)

    best_xgb_r = xg.XGBRegressor(objective='reg:gamma', seed=123, max_leaves=30, learning_rate=0.11, min_child_weight=0.9)
    best_xgb_r.fit(train_X, train_y)

    # Save the model
    with open('xgb_kfold_model.pkl5', 'wb') as f:
        pickle.dump(best_xgb_r, f)

    # Prediction and RMSE for train, validation, and test sets
    train_pred = best_xgb_r.predict(train_X)
    train_rmse = np.sqrt(MSE(train_y, train_pred))
    train_r2 = r2_score(train_y, train_pred)

    test_pred = best_xgb_r.predict(test_X)
    test_rmse = np.sqrt(MSE(test_y, test_pred))
    test_r2 = r2_score(test_y, test_pred)

    # Print RMSE and R2 scores
    print("{},{},{},{},{}".format(person_number, train_rmse, test_rmse, train_r2, test_r2))

    Graphics.plot_prediction_vs_identity_for_person(person_number, test_pred, test_rmse)

    rmse_test.append(test_rmse)


print("")

print("Time taken: {} seconds".format(time.time() - time_start))
print("Average RMSE: ", np.mean(rmse_test))
print("CV RMSE: ", np.std(rmse_test)/np.mean(rmse_test))
print("Standard Deviation RMSE: ", np.std(rmse_test))
print("Variance RMSE: ", np.var(rmse_test))
print("Max RMSE: ", np.max(rmse_test))
print("Min RMSE: ", np.min(rmse_test))
print("Median RMSE: ", np.median(rmse_test))
