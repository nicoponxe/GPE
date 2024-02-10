import numpy as np
from scipy.signal import butter, filtfilt
from pykalman import KalmanFilter

class DataPreprocessor:
    # just a min-max normalization
    # -------------------------------------------------------------------------
    def min_max_normalization(self, df):
        columns_to_scale = ['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']

        for col in columns_to_scale:
            # only apply if not NaN
            if not df[col].isnull().all():
                df[col] = 1 + (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        return df


    # butterworth filter to smooth the data
    # -------------------------------------------------------------------------
    def butterworth_filter(self, df_column, cutoff = 0.05, fs = 30, order = 3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, df_column)

        return y

    #  kalman filter to smooth the data for the Shank and Thigh Angles
    # -------------------------------------------------------------------------
    def kalman_filter(self, df):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        # columns_to_filter = ['ShankAngles', 'ShankAngularVelocity', 'ThighAngles', 'ThighAngularVelocity']
        columns_to_filter = ['ShankAngles', 'ThighAngles'] # This one works better
        for col in columns_to_filter:
            shank_angles = df[col]
            (filtered_state_means, filtered_state_covariances) = kf.filter(shank_angles)
            new_col = col
            df[new_col] = filtered_state_means.flatten()

        return df
