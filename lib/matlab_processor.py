import os
import re
import csv
import logging
import pandas as pd
import numpy as np

from scipy.io import loadmat
from lib.gpe_math import GPEMath

class MatlabProcessor:
    #  Saves a file to CSV
    # -----------------------------------------------------------------------------------
    def save_to_csv(file):
        # Setting the matlab variables
        mat_file = file
        mat_data = loadmat(mat_file)

        # CSV TEMPORAL
        csv_filename = os.path.basename(mat_file) + '.csv'
        open(csv_filename, "w").close()

        # Checking matlab columns
        for col in mat_data.keys():
            # Only process the non-administrative column (non __)
            if not re.search("__", col):
                with open(csv_filename, 'w', newline='') as csv_file:
                    logging.info('Processing {} converting into {}'.format(mat_file, csv_filename))
                    writer = csv.writer(csv_file)
                    subjects = []
                    for subject_id in range(0, 21):
                        subjects.append("subject_{}".format(subject_id))
                    writer.writerow(subjects)
                    for row in mat_data[col]:
                        writer.writerow(row)
        dataset = pd.read_csv(csv_filename)


        os.remove(csv_filename)  #remove if want to delete
        return dataset

    def to_df(file_name):
        path_to_mat = os.environ['PATH_TO_MAT']

        if(path_to_mat != '' and path_to_mat != None):
            file_name = "{}/{}".format(path_to_mat, file_name)

        dataset = MatlabProcessor.save_to_csv(file_name)

        # Extract name and velocity from the file name
        name = 'Shank' if 'Shk' in file_name else 'Thigh'
        velocity = float(file_name.split('_')[1].split('ms')[0]) / 10

        # Melt the dataframe
        melted_df = dataset.melt(ignore_index=False, var_name='subjects', value_name='values')

        # Add velocity column
        # melted_df[name + 'Velocity'] = velocity

        # Reset index
        melted_df.reset_index(inplace=True)

        # Rename columns
        melted_df.rename(columns={'index': 'gait_percentage'}, inplace=True)

        # Add 1 to the 'index_original' column to start the index from 1
        melted_df['gait_percentage'] = melted_df['gait_percentage'] + 1

        delta_t = 1  # provided by team

        # Calculate angular velocity for each group
        angles = melted_df['values'].to_numpy()
        angular_velocity = [GPEMath.calculate_angular_velocity(angles[i:i + 100], delta_t) for i in range(0, len(angles), 100)]
        angular_velocity = np.concatenate(angular_velocity)
        melted_df[name + 'AngularVelocity'] = angular_velocity

        # integral
        integral = [GPEMath.calculate_integral(angles[i:i + 100], delta_t, velocity) for i in range(0, len(angles), 100)]
        melted_df[name + 'Integral'] = np.concatenate(integral)  # Updated this line

        df = melted_df.rename(columns={'values': name + 'Angles'})
        # Reorder the DataFrame columns
        df = df[['gait_percentage', name + 'Angles', name + 'AngularVelocity']]

        return df
