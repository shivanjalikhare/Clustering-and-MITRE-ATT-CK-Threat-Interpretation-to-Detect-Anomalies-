# correlation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationMatrix:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_correlation_matrix(self):
        plt.figure(figsize=(20, 20))
        cor = self.df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
        plt.show()

    def features_with_correlation(self, threshold: float):
        correlated_features = []
        corr_matrix = self.df.corr()

        for column in corr_matrix.columns:
            for other_column in corr_matrix.columns:
                if column != other_column and abs(corr_matrix.loc[column, other_column]) >= threshold:
                    correlated_features.append((column, other_column, corr_matrix.loc[column, other_column]))

        return correlated_features

    def remove_highly_correlated_features(self, threshold: float):
        col_corr = set()
        corr_matrix = self.df.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)

        self.df = self.df.drop(col_corr, axis=1)
        print(f"Removed columns with correlation above {threshold}: {col_corr}")
        return self.df
