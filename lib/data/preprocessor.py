import numpy as np
from scipy.signal import butter, filtfilt
from filterpy.kalman import KalmanFilter

class DataPreprocessor:
    # just a min-max normalization
    # -------------------------------------------------------------------------
    def min_max_normalization(self, df, columns_to_scale):
        for col in columns_to_scale:
            # only apply if not NaN
            if not df[col].isnull().all():
                df[col] = 1 + (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        return df


    # butterworth filter to smooth the data [deprecated]
    # -------------------------------------------------------------------------
    def butterworth_filter(self, df_column, cutoff = 0.05, fs = 30, order = 3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, df_column)

        return y

    #  kalman filter to smooth the data for the columns specified
    # -------------------------------------------------------------------------
    def kalman_filter(self, df, columns_to_filter):
        for col in columns_to_filter:
            df[col + 'Original'] = df[col]

            kf = KalmanFilter(dim_x=1, dim_z=1)
            kf.x = np.array([[0.]])
            kf.P = np.array([[1.]])
            kf.F = np.array([[1.]])
            kf.H = np.array([[1.]])
            kf.Q = np.array([[0.001]])
            kf.R = np.array([[0.1]])

            filtered_values = []
            for measurement in df[col]:
                kf.predict()
                kf.update(measurement)
                filtered_values.append(kf.x[0, 0])

            df[col] = filtered_values

        return df
