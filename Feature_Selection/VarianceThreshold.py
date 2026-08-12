# variance_threshold.py

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class VarianceThresholdSelection:
    def __init__(self, df: pd.DataFrame, threshold: float = 0.5):
        self.df = df
        self.threshold = threshold

    def apply_variance_threshold(self):
        var_thres = VarianceThreshold(threshold=self.threshold)

        var_thres.fit(self.df)

        variances = var_thres.variances_

        variance_df = pd.DataFrame({
            'Column': self.df.columns,
            'Variance': variances
        })

        print(f"Variance of features:\n{variance_df}\n")

        columns_to_remove = variance_df[variance_df['Variance'] < self.threshold]['Column']
        print(f"Columns with variance below {self.threshold} that will be removed:\n{columns_to_remove}\n")

        self.df = self.df.drop(columns=columns_to_remove)

        return self.df

