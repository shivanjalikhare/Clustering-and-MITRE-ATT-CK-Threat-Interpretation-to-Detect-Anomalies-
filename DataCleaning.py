import ipaddress

import pandas as pd

class DataCleaning:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_garbage_rows(self, column_name: str):
        self.df = self.df[self.df[column_name] != column_name].reset_index(drop=True)
        return self

    def remove_garbage_columns(self, columns: list):
        self.df.drop(columns=columns, inplace=True, errors='ignore')
        return self

    def convert_to_standard_data_types(self, conversions: dict):
        for col, dtype in conversions.items():
            if dtype == 'numeric':
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except ValueError:
                    print(f"Warning: Could not convert column '{col}' to type {dtype}.")
                    continue
            elif dtype == 'str':
                try:
                    self.df[col] = self.df[col].astype(str)
                except ValueError:
                    print(f"Warning: Could not convert column '{col}' to type {dtype}.")
                    continue

        return self

    def hex_to_decimal(self, hex_columns: list):
        for col in hex_columns:
            self.df[col] = self.df[col].apply(
                lambda x: int(str(x), 16) if pd.notna(x) and isinstance(x, str) else x
            )
        return self

    def remove_constant_columns(self):
        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) == 1]
        self.df.drop(columns=constant_cols, inplace=True)
        return self

    def fill_na_values(
            self,
            numeric_fill: dict = None,
            categorical_fill: dict = None,
            conditional_replace: list = None,
            fallback_fill: list = None
    ):

        # 1. Fill numeric columns with specified values
        if numeric_fill:
            for col, value in numeric_fill.items():
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(value)

        # 2. Fill categorical columns with specified values
        if categorical_fill:
            for col, value in categorical_fill.items():
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(value)

        # 3. Conditional replacement based on a value
        if conditional_replace:
            for rule in conditional_replace:
                col = rule.get('column')
                condition_value = rule.get('if_value')
                source_col = rule.get('from_column')

                if col in self.df.columns and source_col in self.df.columns:
                    self.df[col] = self.df.apply(
                        lambda row: row[source_col] if row[col] == condition_value and pd.notna(row[source_col]) else
                        row[col],
                        axis=1
                    )

        # 4. Fallback fill: If one column's value is NaN, take the value from another column
        if fallback_fill:
            for rule in fallback_fill:
                col = rule.get('column')
                source_col = rule.get('from_column')

                if col in self.df.columns and source_col in self.df.columns:
                    self.df[col] = self.df.apply(
                        lambda row: row[source_col] if pd.isna(row[col]) and pd.notna(row[source_col]) else row[col],
                        axis=1
                    )

        return self

    def categorical_to_numerical(self, columns: dict):
        def ip_to_int(ip):
            return int(ipaddress.IPv4Address(ip))

        def mac_to_int(mac):
            return int(mac.replace(":", ""), 16)

        for col, func in columns.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(func)

        return self

    def get_cleaned_data(self):
        return self.df
