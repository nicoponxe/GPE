import numpy as np
import pandas as pd

class AugmentationMethods:
    def augment_dataset(self, df, include_shank = True, include_thigh = True):
        result = df
        # print("Original Dataset Length: {}".format(len(result)))
        result = AugmentationMethods().time_shift(result, 2, include_shank=include_shank, include_thigh=include_thigh)
        # print("Length after time_shift: {}".format(len(result)))
        return result

    #  Performs a time shift on the dataset
    # -----------------------------------------------------------------------------------
    def time_shift(self, df, n_shifts = 1, include_shank = True, include_thigh = True):
        augmented_dfs = []
        for shift in range(-n_shifts, n_shifts + 1):
            shifted_df = df.copy()
            if include_shank:
                shifted_df['ShankAngles'] = np.roll(shifted_df['ShankAngles'], shift)
            if include_thigh:
                shifted_df['ThighAngles'] = np.roll(shifted_df['ThighAngles'], shift)
            augmented_dfs.append(shifted_df)
        return pd.concat(augmented_dfs, axis=0)