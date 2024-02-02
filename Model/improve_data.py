import numpy as np

class ImproveData:
    # This method will add non-linear data to the dataframe
    # It assumes the dataset has ShankAngles and ThighAngles columns
    def add_non_linear_data(self, df):
        df['non_linear_1'] = np.sqrt(df['ShankAngles'] ** 2 + df['ShankAngularVelocity'] ** 2)
        df['non_linear_2'] = df['ShankAngularVelocity'] / df['ShankAngles']
        return df
