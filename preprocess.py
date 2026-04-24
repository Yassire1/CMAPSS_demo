"""
Preprocessing utilities for NASA C-MAPSS FD001 LSTM model.
Replicates the exact pipeline from:
  CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Parameters extracted from the notebook
WINDOW_LENGTH = 30
SHIFT = 1
EARLY_RUL = 125
NUM_TEST_WINDOWS = 5
COLUMNS_TO_DROP = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]


def process_targets(data_length, early_rul=None):
    """Create RUL target vector."""
    if early_rul is None:
        return np.arange(data_length - 1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length - 1, -1, -1)
        else:
            return np.append(
                early_rul * np.ones(shape=(early_rul_duration,)),
                np.arange(early_rul - 1, -1, -1)
            )


def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    """Generate windowed batches."""
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(
        num_batches, window_length, num_features
    )
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    """Extract last num_test_windows from test engine."""
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift
        )
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(
            test_data_for_an_engine[-required_len:, :],
            target_data=None,
            window_length=window_length,
            shift=shift
        )
        return batched_test_data_for_an_engine, num_test_windows


def load_and_preprocess(train_path, test_path, rul_path):
    """
    Load raw CMAPSS files and apply the FD001 LSTM preprocessing pipeline.
    Returns:
        processed_test_data: np.array shape (num_test_engines, 30, 14)
        true_rul: np.array shape (100,)
        scaler: fitted StandardScaler
        test_df_raw: raw test DataFrame (for UI display)
    """
    train_data = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_data = pd.read_csv(test_path, sep=r"\s+", header=None)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None)

    # Keep first column (engine id) before scaling
    train_data_first_column = train_data[0]
    test_data_first_column = test_data[0]

    # Global StandardScaler (as used in the LSTM notebook)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.drop(columns=COLUMNS_TO_DROP))
    test_data_scaled = scaler.transform(test_data.drop(columns=COLUMNS_TO_DROP))

    train_data = pd.DataFrame(data=np.c_[train_data_first_column, train_data_scaled])
    test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data_scaled])

    num_test_machines = len(test_data[0].unique())

    processed_test_data = []
    num_test_windows_list = []

    for i in np.arange(1, num_test_machines + 1):
        temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values

        if len(temp_test_data) < WINDOW_LENGTH:
            raise AssertionError(
                f"Test engine {i} doesn't have enough data for window_length of {WINDOW_LENGTH}"
            )

        test_data_for_an_engine, num_windows = process_test_data(
            temp_test_data,
            window_length=WINDOW_LENGTH,
            shift=SHIFT,
            num_test_windows=NUM_TEST_WINDOWS
        )
        processed_test_data.append(test_data_for_an_engine)
        num_test_windows_list.append(num_windows)

    processed_test_data = np.concatenate(processed_test_data)
    true_rul = true_rul[0].values

    return processed_test_data, true_rul, scaler, num_test_windows_list
