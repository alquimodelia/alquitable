import math

import keras
import numpy as np
import pandas as pd

# Create a generator for the timeseries with a moving window.


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset: pd.DataFrame,
        # batch size 7 days, 168 hours,
        time_moving_window_size_X=7 * 24,
        # to predict the after 1 day, 24 hours,
        time_moving_window_size_Y=1 * 24,
        y_columns=None,
        keep_y_on_x=True,
        drop_cols="datetime",
        train_features_folga=0,  # 24 horas de diferenca DA
        skiping_step=1,
        time_cols=None,
        phased_out_columns=None,
        label_col=None,
        alloc_column=None,
        commun_timesteps=0,
    ):
        if alloc_column is None:
            alloc_column = []
        if phased_out_columns is None:
            phased_out_columns = []
        if time_cols is None:
            time_cols = []
        if y_columns is None:
            y_columns = []
        self.train_features_folga = train_features_folga
        self.skiping_step = skiping_step
        self.commun_timesteps = commun_timesteps

        # Make the y the 1st column
        dataset = dataset[
            y_columns
            + [col for col in dataset.columns if col not in y_columns]
        ]

        # make the time columns the last
        dataset = dataset[
            [col for col in dataset.columns if col not in time_cols]
            + time_cols
        ]

        if drop_cols:
            dataset = dataset.drop(drop_cols, axis=1)
        if len(phased_out_columns) > 0:
            dataset[phased_out_columns] = dataset[phased_out_columns].shift(
                train_features_folga
            )
        dataset.dropna(inplace=True)
        self.y_columns = y_columns
        self.y = dataset[self.y_columns].to_numpy()
        if not keep_y_on_x:
            self.x = dataset.loc[:, ~dataset.columns.isin(self.y_columns)]
            self.target_dimension_on_x = [
                self.x.columns.get_loc(c) for c in y_columns if c in self.x
            ]
            self.target_allocated_dimension_on_x = [
                self.x.columns.get_loc(c) for c in alloc_column if c in self.x
            ]
            self.x = self.x.to_numpy()
        else:
            self.x = dataset.to_numpy()
            self.target_dimension_on_x = [
                dataset.columns.get_loc(c) for c in y_columns if c in dataset
            ]
            self.target_allocated_dimension_on_x = [
                dataset.columns.get_loc(c)
                for c in alloc_column
                if c in dataset
            ]

        self.x_batch = time_moving_window_size_X
        self.y_batch = time_moving_window_size_Y + max(
            0, self.commun_timesteps
        )

        self.dataset_size = len(dataset)

    def __len__(self):
        y_alloc = self.y_batch
        if self.commun_timesteps < 0:
            y_alloc = y_alloc + abs(self.commun_timesteps)
        total_batches = self.dataset_size - sum([self.x_batch, y_alloc]) + 1
        total_size = int(math.ceil(total_batches / self.skiping_step))
        return total_size

    def __getitem__(self, index):
        ind = index * self.skiping_step
        limit_point = ind + self.x_batch
        y_limti_point = limit_point - self.commun_timesteps
        X = self.x[ind:limit_point]
        Y = self.y[y_limti_point : y_limti_point + self.y_batch]

        return X, Y


def get_dataset(
    dataset,
    time_moving_window_size_X=168,  # batch size 7 days, 168 hours,
    time_moving_window_size_Y=24,  # to predict the after 1 day, 24 hours,
    y_columns=None,
    keep_y_on_x=True,
    frac=0.9,
    drop_cols="datetime",
    train_features_folga=0,
    skiping_step=1,
    time_cols=None,
    phased_out_columns=None,
    label_col="labels",
    alloc_column=None,
):
    if alloc_column is None:
        alloc_column = ["SecondaryReserveAllocationAUpward"]
    if phased_out_columns is None:
        phased_out_columns = [
            "UpwardUsedSecondaryReserveEnergy",
            "DownwardUsedSecondaryReserveEnergy",
        ]
    if time_cols is None:
        time_cols = []
    if y_columns is None:
        y_columns = []
    if label_col not in dataset:
        label_col = None
    gen = DataGenerator(
        dataset,
        time_moving_window_size_X,
        time_moving_window_size_Y,
        y_columns,
        keep_y_on_x,
        drop_cols=drop_cols,
        train_features_folga=train_features_folga,
        skiping_step=skiping_step,
        time_cols=time_cols,
        phased_out_columns=phased_out_columns,
        label_col=label_col,
        alloc_column=alloc_column,
    )

    X, Y = [], []
    for x, y in gen:
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    train_len = math.ceil(frac * len(X))
    test_len = len(X) - train_len

    train_dataset_X = X[:train_len]
    test_dataset_X = X[train_len : train_len + test_len]

    train_dataset_Y = Y[:train_len]
    test_dataset_Y = Y[train_len : train_len + test_len]

    return (
        train_dataset_X,
        train_dataset_Y,
        test_dataset_X,
        test_dataset_Y,
        gen,
    )
