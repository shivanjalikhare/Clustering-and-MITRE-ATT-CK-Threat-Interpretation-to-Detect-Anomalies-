# feature_selection.py

from CorrelationMatrix import CorrelationMatrix
from ShannonEntropy import ShannonEntropy
from VarianceThreshold import VarianceThresholdSelection


class FeatureSelection:
    def __init__(self, df: pd.DataFrame, corr_threshold: float = 0.9, entropy_threshold: float = 0.25,
                 var_threshold: float = 0.5):
        self.df = df
        self.corr_threshold = corr_threshold
        self.entropy_threshold = entropy_threshold
        self.var_threshold = var_threshold

    def perform_feature_selection(self):
        print("Starting feature selection process...\n")

        correlation_matrix = CorrelationMatrix(self.df)
        correlation_matrix.plot_correlation_matrix()
        correlated_features = correlation_matrix.features_with_correlation(self.corr_threshold)
        print(f"Highly correlated features (correlation > {self.corr_threshold}): {correlated_features}")

        self.df = correlation_matrix.remove_highly_correlated_features(self.corr_threshold)

        shannon_entropy = ShannonEntropy(self.df)
        self.df = shannon_entropy.remove_features_by_entropy()

        variance_threshold = VarianceThresholdSelection(self.df, self.var_threshold)
        self.df = variance_threshold.apply_variance_threshold()

        return self.df
