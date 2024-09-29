import logging
import os
from datetime import datetime

import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Classes.dl_dataset_manager import DLManager, DaysAliveAtHomeRecord, MortalityRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/main.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def identify_non_numeric_columns(array):
    return [
        i for i, col in enumerate(array.T) if not np.all(np.vectorize(is_numeric)(col))
    ]


def is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def replace_non_numeric_with_mean(column):
    numeric_values = [float(val) for val in column if is_numeric(val)]
    if numeric_values:
        mean = np.mean(numeric_values)
    else:
        mean = 0

    return np.array([float(val) if is_numeric(val) else mean for val in column])


def prepare_data(records):
    X_bxb_list = []
    X_cat_list = []
    y_list = []

    for record in records:
        X_bxb = record.x_bxb.values
        X_bxb_list.append(X_bxb)

        X_cat = record.x_cat.values.flatten()
        X_cat_list.append(X_cat)

        y_list.append(record.y)

    X_bxb_array = np.array(X_bxb_list)
    X_cat_array = np.array(X_cat_list)
    y_array = np.array(y_list)

    le = LabelEncoder()

    X_cat_array[:, 4] = le.fit_transform(X_cat_array[:, 4])
    X_cat_array[:, 11] = le.fit_transform(X_cat_array[:, 11])
    X_cat_array[:, 11] = le.fit_transform(X_cat_array[:, 11])
    X_cat_array[:, 15] = le.fit_transform(X_cat_array[:, 15])
    X_cat_array[:, 16] = le.fit_transform(X_cat_array[:, 16])

    def to_unix_timestamp_ms(date_str, column_index):
        try:
            if date_str and date_str.strip() and date_str != "\\":
                if column_index == 13:
                    dt = datetime.strptime(date_str, "%d/%m/%Y")
                    return int(dt.timestamp() * 1000)
                elif column_index == 14:
                    return int(float(date_str) / 1e6)
            else:
                logging.warning(
                    f"Empty or invalid date string in column {column_index}: '{date_str}'"
                )
                return 0
        except ValueError as e:
            logging.error(
                f"Malformed date string in column {column_index}: '{date_str}'. Error: {str(e)}"
            )
            return 0

    X_cat_array[:, 13] = np.array(
        [to_unix_timestamp_ms(str(x), 13) for x in X_cat_array[:, 13]]
    )
    X_cat_array[:, 14] = np.array(
        [to_unix_timestamp_ms(str(x), 14) for x in X_cat_array[:, 14]]
    )

    X_cat_array = np.delete(X_cat_array, 10, 1)

    non_numeric_cols = identify_non_numeric_columns(X_cat_array)
    print(f"Columns with non-numeric data: {non_numeric_cols}")

    X_cat_array = np.apply_along_axis(replace_non_numeric_with_mean, 0, X_cat_array)

    X_cat_array = X_cat_array.astype(float)
    normalizer_cat = MinMaxScaler()

    scaler = StandardScaler()
    X_cat_normalized = scaler.fit_transform(X_cat_array)
    X_cat_normalized = normalizer_cat.fit_transform(X_cat_array)

    if len(X_bxb_array.shape) == 3:
        num_samples, num_timesteps, num_features = X_bxb_array.shape
    else:
        num_samples, num_features = X_bxb_array.shape
        num_timesteps = 1
        X_bxb_array = X_bxb_array.reshape(num_samples, num_timesteps, num_features)

    return X_bxb_array, X_cat_normalized, y_array


def load_and_prepare_data(days):
    mortality_manager = DLManager[MortalityRecord](MortalityRecord, days=days)
    records = mortality_manager.load_mortality_data()
    return prepare_data(records)


def split_and_save_data(
    X_bxb, X_cat, y, output_folder, test_size=0.2, val_size=0.2, random_state=42
):
    X_bxb_train, X_bxb_test, X_cat_train, X_cat_test, y_train, y_test = (
        train_test_split(
            X_bxb, X_cat, y, test_size=test_size, random_state=random_state
        )
    )

    X_bxb_train, X_bxb_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_bxb_train, X_cat_train, y_train, test_size=val_size, random_state=random_state
    )

    np.save(os.path.join(output_folder, "X_bxb_train.npy"), X_bxb_train)
    np.save(os.path.join(output_folder, "X_bxb_test.npy"), X_bxb_test)
    np.save(os.path.join(output_folder, "X_cat_train.npy"), X_cat_train)
    np.save(os.path.join(output_folder, "X_cat_test.npy"), X_cat_test)
    np.save(os.path.join(output_folder, "y_train.npy"), y_train)
    np.save(os.path.join(output_folder, "y_test.npy"), y_test)
    np.save(os.path.join(output_folder, "X_bxb_val.npy"), X_bxb_val)
    np.save(os.path.join(output_folder, "X_cat_val.npy"), X_cat_val)
    np.save(os.path.join(output_folder, "y_val.npy"), y_val)

    return (
        X_bxb_train,
        X_bxb_test,
        X_cat_train,
        X_cat_test,
        y_train,
        y_test,
        X_bxb_val,
        X_cat_val,
        y_val,
    )


def load_and_prepare_daoh_data(days):
    daoh_manager = DLManager[DaysAliveAtHomeRecord](DaysAliveAtHomeRecord, days=days)
    records = daoh_manager.load_days_alive_at_home_data()
    return prepare_data(records)


def generate_mortality_data(days=365, output_folder=None):
    X_bxb, X_cat, y = load_and_prepare_data(days)
    logger.info(f"Breath-by-breath data shape: {X_bxb.shape}")
    logger.info(f"Categorical data shape: {X_cat.shape}")
    logger.info(f"Output shape: {y.shape}")

    if output_folder is None:
        output_folder = os.path.join("data", "ml_inputs", f"mortality_{days}")
    os.makedirs(output_folder, exist_ok=True)

    split_data = split_and_save_data(X_bxb, X_cat, y, output_folder)
    logger.info(f"Data has been split and saved in {output_folder}.")
    return split_data


def generate_days_alive_at_home_data(days=30, output_folder=None):
    X_bxb, X_cat, y = load_and_prepare_daoh_data(days)
    logger.info(f"Breath-by-breath data shape: {X_bxb.shape}")
    logger.info(f"Categorical data shape: {X_cat.shape}")
    logger.info(f"Output shape: {y.shape}")

    if output_folder is None:
        output_folder = os.path.join("data", "ml_inputs", f"daoh_{days}")
    os.makedirs(output_folder, exist_ok=True)

    split_data = split_and_save_data(X_bxb, X_cat, y, output_folder)
    logger.info(f"Data has been split and saved in {output_folder}.")
    return split_data

if __name__ == "__main__":
    generate_mortality_data()
