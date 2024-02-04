import numpy as np

class Kfold:
    def remove_k_persons(self, df, k):
        offset = (k-1)*100   # 100 is the number of percentage points
        removed_person = df.iloc[offset:offset+100].copy()
        new_df = df.drop(df.index[offset:offset+100])

        return new_df, removed_person
