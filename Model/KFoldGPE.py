import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xg
from mat_to_csv import MatToCsv
from improve_data import ImproveData
from kfold import Kfold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score
from augmentation_methods import AugmentationMethods

def calculate_angular_velocity(angles, delta_t):
    return np.concatenate(([np.nan], np.diff(angles) / delta_t))

def calculate_integral(angles, delta_t, velocity):
    integral_values = []
    integral_sum = 0

    for i in range(len(angles) - 1):
        integral_sum += velocity * (angles[i + 1] + angles[i]) * delta_t
        integral_values.append(integral_sum)

    # Add the initial condition (0) to the beginning of the integral_values list
    integral_values.insert(0, 0)

    return integral_values

def preprocess_dataset(file_name):
    matToCsv = MatToCsv()
    path_to_mat = os.environ['PATH_TO_MAT']

    if(path_to_mat != '' and path_to_mat != None):
        file_name = "{}/{}".format(path_to_mat, file_name)

    dataset = matToCsv.dataset(file_name)

    # Extract name and velocity from the file name
    name = 'Shank' if 'Shk' in file_name else 'Thigh'
    velocity = float(file_name.split('_')[1].split('ms')[0]) / 10

    # Melt the dataframe
    melted_df = dataset.melt(ignore_index=False, var_name='subjects', value_name='values')

    # Add velocity column
    #melted_df[name + 'Velocity'] = velocity

    # Reset index
    melted_df.reset_index(inplace=True)

    # Rename columns
    melted_df.rename(columns={'index': 'gait_percentage'}, inplace=True)

    # Add 1 to the 'index_original' column to start the index from 1
    melted_df['gait_percentage'] = melted_df['gait_percentage'] + 1

    delta_t = 1  # provided by team

    # Calculate angular velocity for each group
    angles = melted_df['values'].to_numpy()
    angular_velocity = [calculate_angular_velocity(angles[i:i + 100], delta_t) for i in range(0, len(angles), 100)]
    angular_velocity = np.concatenate(angular_velocity)
    melted_df[name + 'AngularVelocity'] = angular_velocity

    # integral
    integral = [calculate_integral(angles[i:i + 100], delta_t, velocity) for i in range(0, len(angles), 100)]
    melted_df[name + 'Integral'] = np.concatenate(integral)  # Updated this line

    df = melted_df.rename(columns={'values': name + 'Angles'})
    # Reorder the DataFrame columns
    df = df[['gait_percentage', name + 'Angles', name + 'AngularVelocity']]

    return df

print("Person,Train RMSE,Test RMSE,Train R2,Test R2")
# Process all people
for person_number in range(1, 22):

    file_names = ['ShkAngW_05ms.mat', 'ShkAngW_10ms.mat','ShkAngW_15ms.mat', 'ThiAngW_05ms.mat', 'ThiAngW_10ms.mat', 'ThiAngW_15ms.mat']
    datasets = [preprocess_dataset(file_name) for file_name in file_names]

# DataFrame concatenation
    shankDF = pd.concat(datasets[:3], axis=0)
    thighDF = pd.concat(datasets[3:], axis=0).drop('gait_percentage', axis=1)

    k_fold_df = pd.concat([shankDF, thighDF], axis=1)

    k_fold_df, k_fold_person_df  = Kfold().remove_k_persons(k_fold_df, person_number)
    k_fold_df = ImproveData().add_non_linear_data(k_fold_df)       # Suggested by Mahdy
    k_fold_df = AugmentationMethods().augment_dataset(k_fold_df)   # Adds Augmentation methods to the dataset

    k_fold_person_df = ImproveData().add_non_linear_data(k_fold_person_df)

    train_X = k_fold_df[['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']]
    test_X = k_fold_person_df[['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']]

    if('non_linear_1' in k_fold_df.columns):
        train_X = k_fold_df[['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity', 'non_linear_1', 'non_linear_2']]
        test_X = k_fold_person_df[['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity', 'non_linear_1', 'non_linear_2']]

    train_y = k_fold_df['gait_percentage']
    test_y = k_fold_person_df['gait_percentage']

    best_xgb_r = xg.XGBRegressor(objective='reg:gamma', seed=123)
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

# Create a DataFrame with the predicted and original values for ShankAngles and ThighAngles
    df = pd.DataFrame({'predicted_person': test_pred, 'identity': range(1, 101)})

# Create a scatter plot with predicted and original values on the x-axis, and ShankAngles and ThighAngles on the y-axis
    plt.scatter(df['identity'], df['identity'], label='Identity Function'.format(person_number), alpha=0.3, marker='o', s=40, color='green', edgecolors='k')
    plt.scatter(df['identity'], df['predicted_person'], label='Predicted Gait % for Person {}'.format(person_number), alpha=0.7, marker='o', s=40, color='orange', edgecolors='k')

    plt.ylabel('Gait %')
    plt.xlabel('Predicted Gait %')
    plt.title('Prediction vs Identity for Person {}'.format(person_number))

# Add RMSE to the plot
    plt.text(0.05, 0.95, f"RMSE: {round(test_rmse, 2)}", transform=plt.gca().transAxes)

    plt.legend()
    plt.savefig('person_{}.png'.format(person_number))
    plt.clf() # Clear the plot

