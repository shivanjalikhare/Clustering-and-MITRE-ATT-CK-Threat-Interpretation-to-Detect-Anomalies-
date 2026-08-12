# shannon_entropy.py

import pandas as pd
import numpy as np
from scipy.stats import entropy


class ShannonEntropy:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def calculate_entropy(self, feature):
        value_counts = feature.value_counts(normalize=True)
        return entropy(value_counts)

    def entropy_for_all_features(self):
        entropy_table = []
        for column in self.df.columns:
            column_entropy = self.calculate_entropy(self.df[column].dropna())
            entropy_table.append({'Feature': column, 'Entropy': column_entropy})

        return pd.DataFrame(entropy_table)

    def remove_features_by_entropy(self):
        entropy_table = self.entropy_for_all_features()
        Q1 = np.percentile(entropy_table['Entropy'], 25)

        print(f"First Quartile (Q1) Entropy Threshold: {Q1:.4f}")

        features_to_remove = entropy_table[entropy_table['Entropy'] < Q1]['Feature']

        self.df = self.df.drop(columns=features_to_remove)

        print(f"Removed features with entropy less than Q1: {list(features_to_remove)}")
        return self.df
