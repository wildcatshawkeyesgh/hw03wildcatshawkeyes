import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


__all__ = ["DataProcessor", "DataPrep"]


class DataProcessor:
    wheel_speed_file = "_decoded_wheel_speed_fl.csv"
    status_file = "_acc_status.csv"
    kmh_ms_conversion = 1 / 3.6
    num_lags = 9

    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)

    def find_matching_files(self):
        speed_files = list(self.input_folder.glob(f"*{self.wheel_speed_file}"))
        pairs = []

        for speed_file in speed_files:

            prefix = speed_file.name[: -len(self.wheel_speed_file)]
            status_file = self.input_folder / f"{prefix}{self.status_file}"

            if status_file.exists():
                pairs.append((speed_file, status_file))
            else:
                print(f"this file is missing {speed_file.name}")

        return pairs

    def apply_zero_order(self, speed_df, status_df):
        merged = speed_df.join_asof(status_df, on="Time", strategy="backward")
        return merged

    def add_lag_columns(self, df):

        for lag in range(1, self.num_lags + 1):
            column_name = f"speed_{lag}"
            df = df.with_columns(
                pl.col("speed").shift(lag).fill_null(0).alias(column_name)
            )

        return df

    def replace_status(self, df):
        df_replaced = df.with_columns(
            pl.when(pl.col("status") == 6).then(1).otherwise(0).alias("status")
        )
        return df_replaced

    def normalize(self, df):
        mean = df["speed"].mean()
        std = df["speed"].std()
        df = df.with_columns(((pl.col("speed") - mean) / std).alias("speed"))

        return df

    def process_pair(self, speed_file, status_file):

        speed_df = pl.read_csv(speed_file).select(["Time", "Message"])
        status_df = pl.read_csv(status_file).select(["Time", "Message"])

        speed_df = speed_df.rename({"Message": "speed"})
        status_df = status_df.rename({"Message": "status"})

        speed_df = speed_df.with_columns(
            (pl.col("speed") * self.kmh_ms_conversion).alias("speed")
        )
        merged_df = self.apply_zero_order(speed_df, status_df)
        final_df = self.replace_status(merged_df)

        return final_df

    def process_all(self):

        pairs = self.find_matching_files()

        all_dfs = []

        for speed_file, status_file in pairs:
            result = self.process_pair(speed_file, status_file)
            all_dfs.append(result)

        combined = pl.concat(all_dfs)
        print(combined["status"].value_counts())
        combined = self.normalize(combined)
        combined = self.add_lag_columns(combined)

        output_path = self.output_folder / "Final.csv"
        combined.write_csv(output_path)
        print(f"Saved final csv to {output_path}")

        return output_path


class DataPrep:

    def __init__(self, data_path, batch_size):
        self.data_path = Path(data_path)
        self.batch_size = batch_size

    def get_features_results(self):
        self.df = pl.read_csv(self.data_path)

        feature_cols = ["speed"] + [f"speed_{i}" for i in range(1, 10)]
        features = self.df.select(feature_cols)
        results = self.df.select(["status"])

        return features, results

    def get_train_test(self, features, results):
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, results, test_size=0.2
        )
        return X_train, X_test, Y_train, Y_test

    def dataload(self):
        features, results = self.get_features_results()
        X_train, X_test, Y_train, Y_test = self.get_train_test(features, results)

        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.long).squeeze()
        Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.long).squeeze()

        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)

        self.train_loader = TorchDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = TorchDataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
