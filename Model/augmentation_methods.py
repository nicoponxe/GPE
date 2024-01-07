import numpy as np
import pandas as pd

class AugmentationMethods:
    def augment_dataset(self, df):
        result = df
        print("Original Dataset Length: {}".format(len(result)))
        result = AugmentationMethods().time_shift(result)
        print("Length after time_shift: {}".format(len(result)))
        return result

    def time_shift(self, df, n_shifts = 1):
        augmented_dfs = []
        for shift in range(-n_shifts, n_shifts + 1):
            shifted_df = df.copy()
            shifted_df['ShankAngles'] = np.roll(shifted_df['ShankAngles'], shift)
            shifted_df['ThighAngles'] = np.roll(shifted_df['ThighAngles'], shift)
            augmented_dfs.append(shifted_df)
        return pd.concat(augmented_dfs, axis=0)
