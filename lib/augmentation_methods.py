import numpy as np
import pandas as pd

class AugmentationMethods:
    def augment_dataset(self, df, include_shank = True, include_thigh = True, n_shifts = 1):
        result = df
        # print("Original Dataset Length: {}".format(len(result)))
        result = AugmentationMethods().time_shift(result, n_shifts = n_shifts, include_shank=include_shank, include_thigh=include_thigh)
        # print("Length after time_shift: {}".format(len(result)))
        return result

    #  Performs a time shift on the dataset
    # -----------------------------------------------------------------------------------
    def time_shift(self, df, n_shifts = 1, include_shank = True, include_thigh = True):
        # No data augmentation required
        if n_shifts == 0:
            return df

        augmented_dfs = []
        for shift in range(-n_shifts, n_shifts + 1):
            shifted_df = df.copy()
            if include_shank:
                shifted_df['ShankAngles'] = np.roll(shifted_df['ShankAngles'], shift)
                shifted_df['ShankAngularVelocity'] = np.roll(shifted_df['ShankAngularVelocity'], shift)
            if include_thigh:
                shifted_df['ThighAngles'] = np.roll(shifted_df['ThighAngles'], shift)
                shifted_df['ThighAngularVelocity'] = np.roll(shifted_df['ThighAngularVelocity'], shift)
            augmented_dfs.append(shifted_df)
        return pd.concat(augmented_dfs, axis=0)
